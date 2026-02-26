"""
poker_env.py
------------
A Gymnasium-compatible environment that wraps PokerKit's
NoLimitTexasHoldem into a single-agent RL loop.

                ┌──────────────────────────────────┐
                │         PokerEnv                 │
                │                                  │
   reset() ───► │  deal cards                      │
                │  return obs, info                │
                │                                  │
   step(act) ─► │  validate action (legal mask)    │
                │  execute action via PokerKit     │
                │  if hand over → compute reward   │
                │  return obs, reward, done, info  │
                └──────────────────────────────────┘

Design notes
────────────
* The "agent" is always player 0.  Player 1 is a simple random
  opponent so we have something to train against.
* Observation vector (length 54):
      [0 .. 16]   – 17-length one-hot card encoding of hole cards
      [17 .. 33]  – 17-length one-hot card encoding of community cards
      [0-1, 0,0,0,0] 0-1 normalized value for rank / 12 and one hot vector for suit
      [34]        - AGENT position (0 sb, 1 bb)
      [35]        – agent's normalised stack
      [36]        – opponent's normalised stack
      [37]        – normalised pot size
      [38 - 41]   – which betting round one hot encoded (0-1)
      [42 .. 47]  – opponent's last action (one-hot, 6 actions)
      [48 .. 53]  – number of community cards dealt (one-hot, 0-5)

* Action space: Discrete(6)  →  fold / check / call / raise_quarter /
                                 raise_half / raise_pot
* A legal-action mask is passed inside `info` every step so the PPO
  network can zero-out illegal moves before sampling.
"""

import random
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pokerkit import Automation, NoLimitTexasHoldem

from bot import PokerBot

# ── card encoding helpers ────────────────────────────────────────────
RANKS = "23456789TJQKA"
SUITS = "cdhs"


RANK_TO_IDX: dict[str, int] = {
    r: i + 1 for i, r in enumerate(RANKS)
    # for j, s in enumerate(SUITS)
}  # e.g. "2c"->0, "2d"->1, … "As"->51

SUIT_TO_IDX: dict[str, int] = {
    s: j for j, s in enumerate(SUITS)
}  # e.g. "c"->0, "d"->1, … "s"->3

# print(RANK_TO_IDX)
# print(SUIT_TO_IDX)
def encode_card(card: str) -> np.ndarray:
    """Return a 5-dim hybrid encoding vector suit/rank for a list of card strings."""
    vec = np.zeros(5, dtype=np.float32)
    rank, suit = card[0], card[1]
    vec[0] = RANK_TO_IDX[rank] / 13.0  # Normalize rank to [0,1]
    vec[1 + SUIT_TO_IDX[suit]] = 1.0
    return vec

def cards_to_vector(cards: list[str]) -> np.ndarray:
    # print("cards:",     cards)
    vectors = [encode_card(c) for c in cards]
    if len(vectors) == 0:
        return np.zeros(5, dtype=np.float32)
    return np.concatenate(vectors)



# ── environment ──────────────────────────────────────────────────────
class PokerEnv(gym.Env):
    """
    Single-agent heads-up No-Limit Texas Hold'em environment.

    Parameters
    ----------
    buy_in   : float   – starting stack for both players
    sb       : float   – small blind
    bb       : float   – big blind
    """

    metadata = {"render_modes": []}

    # action indices
    FOLD = 0
    CHECK = 1
    CALL = 2
    RAISE_QUARTER = 3
    RAISE_HALF = 4
    RAISE_POT = 5

    def __init__(self, buy_in: float = 100.0, sb: float = 5.0, bb: float = 10.0):
        super().__init__()
        self.buy_in = buy_in
        self.sb = sb
        self.bb = bb

        # Gymnasium required spaces
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(54,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(PokerBot.NUM_ACTIONS)

        # PokerBot instances (used for reward / logging)
        self.agent = PokerBot("Agent")
        self.opp = PokerBot("Opponent")

        # PokerKit state (initialised in reset)
        self.state: Any = None
        self.done = True
        
        # Track opponent's last action for observation
        self.opp_last_action: int = -1  # -1 = no action yet

    # ── public API ───────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._new_hand()
        obs = self._build_obs()
        info = {"legal_mask": self._legal_mask()}
        return obs, info


    def _hand_strength(self):
        if not self.state.board_cards:
            return 0.0
        
        # get_hand(player_index, board_index, hand_type_index)
        # player_index=0 (agent), board_index=0 (main board), hand_type_index=0 (high hand)
        hand = self.state.get_hand(0, 0, 0)
        
        if hand is None:
            return 0.0

        # Extract hand name from string like 'One pair (AcAdJsTs2c)'
        hand_str = str(hand).split('(')[0].strip().lower()
        
        rank_values = {
            'high card': 0,
            'one pair': 1,
            'two pair': 2,
            'three of a kind': 3,
            'straight': 4,
            'flush': 5,
            'full house': 6,
            'four of a kind': 7,
            'straight flush': 8,
            'royal flush': 9,
        }

        return rank_values.get(hand_str, 0.0) / 9.0

    def step(self, action: int):
        # Track which betting round we're in before the action
        round_before = self._betting_round()

        # Small penalty for folding (to discourage folding when checking is possible)
        # Normalize the penalty to the buy-in so rewards are on a consistent scale
        # fold_penalty = (-0.5 / self.buy_in) if action == self.FOLD else 0.0

        # If it's the opponent's turn first (e.g. pre-flop SB acts
        # first in heads-up), let the random opponent act until it's
        # the agent's turn OR the round changes.
        self._maybe_run_opponent(round_before)

        # Execute the agent's action
        self._execute_action(action, player=0)

        # Deal board cards if a betting round just ended
        self._deal_board_if_needed()

        # Track the new round after dealing
        round_after = self._betting_round()

        # If the hand isn't over, let the opponent respond (but only within same round
        # or until it's agent's turn again)
        if self.state.actor_index != 0 and not self._hand_over():
            self._maybe_run_opponent(round_after)

        # Deal board again in case opponent's actions ended a round
        self._deal_board_if_needed()

        # Check if hand is finished
        reward = 0.0  # Start with zero reward
        done = False
        if self._hand_over():
            reward += self._resolve_hand()  # Add hand result (already normalized)
            done = True
            self.done = True
            # Don't build obs when done - reset() will be called next
            obs = np.zeros(54, dtype=np.float32)
            info = {"legal_mask": [0, 1, 0, 0, 0, 0]}
        else:
            obs = self._build_obs()
            info = {"legal_mask": self._legal_mask()}
        
        truncated = False
        return obs, reward, done, truncated, info

    # ── internal: hand lifecycle ─────────────────────────────────────
    def _new_hand(self):
        """Create a fresh PokerKit state and deal hole cards."""
        # Reset stacks to buy-in for simplicity (one hand per episode)
        self.agent.cash_val = self.buy_in
        self.opp.cash_val = self.buy_in

        self.state = NoLimitTexasHoldem.create_state(
            (
                Automation.ANTE_POSTING,
                Automation.BET_COLLECTION,
                Automation.BLIND_OR_STRADDLE_POSTING,
                Automation.CARD_BURNING,
                Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
                Automation.HAND_KILLING,
                Automation.CHIPS_PUSHING,
                Automation.CHIPS_PULLING,
            ),
            True,  # uniform antes
            0,  # antes (none)
            (self.sb, self.bb),  # blinds
            self.bb,  # min bet
            (self.buy_in, self.buy_in),  # starting stacks
            2,  # num players (heads-up)
        )

        # Deal 2 hole cards to each player
        self.state.deal_hole(self._random_hand())  # player 0 (agent)
        self.state.deal_hole(self._random_hand())  # player 1 (opp)
        self.done = False
        self.opp_last_action = -1  # Reset opponent action tracking

        self.agent_position = self.state.actor_index  # 0=SB, 1=BB in heads-up

    # ── internal: opponent logic ─────────────────────────────────────
    def _maybe_run_opponent(self, current_round: int):
        """
        Let the random opponent act whenever it's their turn.
        Stop if the hand ends, it becomes the agent's turn, or we move to a new round.
        """
        while (
                not self._hand_over()
                and self.state.actor_index == 1
        ):
            opp_action = self._random_action()
            self.opp_last_action = opp_action  # Track opponent's action
            self._execute_action(opp_action, player=1)
            # Deal board cards if a betting round just ended
            self._deal_board_if_needed()
            # If we've moved to a new round, stop and let agent act
            if self._betting_round() != current_round:
                break

    def _random_action(self) -> int:
        """Simple random policy for the opponent."""
        mask = self._legal_mask()
        legal = [i for i, v in enumerate(mask) if v]
        return random.choice(legal)

    # ── internal: action execution ───────────────────────────────────
    def _execute_action(self, action: int, player: int):
        """Translate a discrete action index into a PokerKit call."""
        try:
            if action == self.FOLD:
                self.state.fold()

            elif action == self.CHECK:
                self.state.check_or_call()  # PokerKit unifies check/call

            elif action == self.CALL:
                self.state.check_or_call()

            elif action in (self.RAISE_QUARTER, self.RAISE_HALF, self.RAISE_POT):
                pot = self._current_pot()
                stack = self.state.stacks[player]
                multiplier = {
                    self.RAISE_QUARTER: 0.25,
                    self.RAISE_HALF: 0.50,
                    self.RAISE_POT: 1.00,
                }[action]
                raise_amount = pot * multiplier
                # The amount passed to complete_bet_or_raise_to is the
                # *total* bet size, not the increment.  We add the
                # current bet the player has already put in this round.
                current_bet = self.state.bets[player]
                target = current_bet + raise_amount
                # Clamp to player's stack (all-in)
                target = min(target, stack)
                # PokerKit requires target >= min raise; clamp up
                target = max(target, self.state.min_completion_raising_to_amount or target)
                self.state.complete_bet_or_raise_to(target)

            # After a betting action the board may need dealing (flop/turn/river).
            # PokerKit automations handle card_burning but we must
            # manually deal_board when the phase advances.
            self._deal_board_if_needed()

        except Exception:
            # If the action is illegal PokerKit raises; fall back to check/call
            try:
                self.state.check_or_call()
            except Exception:
                pass  # truly stuck – hand will resolve

    # ── internal: board dealing ──────────────────────────────────────
    def _deal_board_if_needed(self):
        """Deal community cards when PokerKit expects them."""
        if self._hand_over():
            return
        # Keep dealing board cards while PokerKit expects them
        while hasattr(self.state, 'can_deal_board') and self.state.can_deal_board():
            # deal_board expects a string of cards; use random cards
            cards_needed = self.state.board_dealing_count if hasattr(self.state, 'board_dealing_count') else 3
            card_str = self._random_cards(cards_needed)
            self.state.deal_board(card_str)

    def _one_hot_betting_round(self) -> np.ndarray:
        """Return a one-hot vector for the current betting round."""
        round_vec = np.zeros(4, dtype=np.float32)
        round_idx = self._betting_round()
        round_vec[round_idx] = 1.0
        return round_vec

    def _one_hot_opponent_last_action(self) -> np.ndarray:
        """Return a one-hot vector for the opponent's last action."""
        action_vec = np.zeros(6, dtype=np.float32)
        if 0 <= self.opp_last_action < 6:
            action_vec[self.opp_last_action] = 1.0
        return action_vec

    # ── internal: observation ────────────────────────────────────────
    def _build_obs(self) -> np.ndarray:
        obs = np.zeros(54, dtype=np.float32)

        # [0..51] – one-hot cards (hole + board)
        hole_cards: list[str] = []
        community_cards: list[str] = []
        bet_round = self._betting_round()
        # Agent's hole cards
        if self.state.hole_cards and self.state.hole_cards[0]:
            hole_cards += [c.rank+c.suit for c in self.state.hole_cards[0]]
        # Board cards - flatten nested structure (flop, turn, river are separate tuples)
        # board cards: list[list[Card]] ---> [[c1, c2, c3, c4, c5], [c11,c22,c33,c44,c55]]
        if self.state.board_cards and self.state.board_cards:
            community_cards += [c[0].rank+c[0].suit for c in self.state.board_cards]
        # print(visible_cards)
        obs[0:10] = cards_to_vector(hole_cards)
        # print("HOLE CARDS")
        # print("obs:", obs[0:5])
        # print("obs:", obs[5:10])

        board_vec = cards_to_vector(community_cards)



        # size 0, 15, 20, 25
        # Dynamically allocates cards that were dealt in the commmunity
        obs[10:10+len(board_vec)] = board_vec
        # print("BOARD CARDS")
        # for i in range(0, len(board_vec), 5):
        #     print("obs:", obs[10+i:10+i+5])


        # Tracks small and big blind
        obs[35] = self.agent_position  # 0 for SB, 1 for BB 
        # # Tracks agent stack and opponent stack normalised to buy-in
        obs[36] = self.state.stacks[0] / self.buy_in #agent stack
        obs[37] = self.state.stacks[1] / self.buy_in #opponent stack
        # # Tracks pot size normalised to max possible pot 
        obs[38] = self._current_pot() / (2 * self.buy_in) # normalized pot

        # # Hand strength normalised to [0,1] based on hand rank (HIGH_CARD=0, ROYAL_FLUSH=1)
        obs[39] = self._hand_strength()
        # print("hand strength:", obs[39])

        # [40 - 43]   – which betting round one hot encoded (4 dims)
        obs[40:44] = self._one_hot_betting_round()

        betting_rounds = ["preflop", "flop", "turn", "river"]
        # print("betting round:", betting_rounds[bet_round])
        
        # [44 .. 49]  – opponent's last action (one-hot, 6 actions)
        obs[44:50] = self._one_hot_opponent_last_action()

        # [48 .. 53]  – number of community cards dealt (one-hot, 0-5)
        
        # # [52] agent stack (normalised)
        # obs[52] = self.state.stacks[0] / self.buy_in
        # # [53] opponent stack (normalised)
        # obs[53] = self.state.stacks[1] / self.buy_in
        # # [54] pot (normalised)
        # obs[54] = self._current_pot() / (2 * self.buy_in)
        # # [55] betting round  0=preflop 1=flop 2=turn 3=river
        # obs[55] = self._betting_round() / 3.0
        # # [56..61] – opponent last action one-hot (zeros if unknown)
        # # (tracked externally; placeholder here)
        # # [62..66] – number of community cards one-hot
        # n_board = len(self.state.board_cards) if self.state.board_cards else 0
        # if n_board <= 5:
        #     obs[62 + n_board] = 1.0

        return obs

    # ── internal: legal action mask ──────────────────────────────────
    def _legal_mask(self) -> list[int]:
        """
        Return [0 or 1] * 6.  The PPO wrapper uses this to mask
        probabilities before sampling an action.
        """
        mask = [0, 0, 0, 0, 0, 0]
        if self._hand_over():
            mask[self.CHECK] = 1  # dummy; env is done
            return mask

        # Determine what PokerKit allows
        can_call = self.state.can_check_or_call()
        can_raise = self.state.can_complete_bet_or_raise_to()
        # Fold is always legal when it's your turn
        mask[self.FOLD] = 1

        if can_call:
            # If no outstanding bet → check; else → call
            if self.state.bets[0] >= (self.state.bets[1] if len(self.state.bets) > 1 else 0):
                mask[self.CHECK] = 1
            else:
                mask[self.CALL] = 1

        if can_raise:
            mask[self.RAISE_QUARTER] = 1
            mask[self.RAISE_HALF] = 1
            mask[self.RAISE_POT] = 1

        return mask

    # ── internal: hand resolution ────────────────────────────────────
    def _resolve_hand(self) -> float:
        """
        Figure out who won and return the reward for the agent.
        """
        # PokerKit updates stacks automatically via CHIPS_PUSHING
        agent_final = self.state.stacks[0]
        opp_final = self.state.stacks[1]

        # Compute net change relative to starting buy-in and normalize
        net = float(agent_final) - float(self.buy_in)
        # Record absolute wins/losses for logging
        if net > 0:
            self.agent.record_win(net/float(self.buy_in))
        elif net < 0:
            self.agent.record_loss(-net/float(self.buy_in))

        # Return normalized reward (fraction of buy-in)
        return net / float(self.buy_in)

    # ── internal: utility ────────────────────────────────────────────
    def _hand_over(self) -> bool:
        """True when PokerKit considers the hand finished."""
        # PokerKit sets actor_index to None when no one can act
        # BUT the hand isn't truly over if we still need to deal board cards
        if self.state.actor_index is not None:
            return False
        # If actor is None but we can deal board, hand continues
        if hasattr(self.state, 'can_deal_board') and self.state.can_deal_board():
            return False
        return True

    def _current_pot(self) -> float:
        """Total pot amount including collected bets."""
        return float(self.state.total_pot_amount) if hasattr(self.state, 'total_pot_amount') else 0.0

    def _betting_round(self) -> int:
        """0=preflop, 1=flop, 2=turn, 3=river based on board size."""
        n = len(self.state.board_cards) if self.state.board_cards else 0
        if n == 0:
            return 0
        elif n == 3:
            return 1
        elif n == 4:
            return 2
        else:
            return 3


    def _random_hand(self) -> str:
        """Generate a random 2-card hand string like 'AhKs'."""
        dealable = list(self.state.get_dealable_cards())
        # print(dealable)
        cards = random.sample(dealable, 2)
        # print(cards)
        return "".join(c.rank + c.suit for c in cards)

    def _random_cards(self, n: int) -> str:
        """Generate n random cards as a single string."""
        dealable = list(self.state.get_dealable_cards())
        # print(dealable)
        cards = random.sample(dealable, n)
        # print(cards)
        return "".join(c.rank + c.suit for c in cards)


if __name__ == "__main__":
    env = PokerEnv(buy_in=100.0, sb=5.0, bb=10.0)
    env._new_hand()
    
    # print("Hole cards:", [str(c) for c in env.state.hole_cards[0]])
    # print("Board cards:", [str(c) for c in env.state.board_cards] if env.state.board_cards else [])
    
    
    # obs = env._build_obs()
    # # print("Observation:", obs)
    # print("strength:", obs[39])

    # print("Testing complete!")
    # print(f"{'='*50}")

    # print("Street index:", env.state.street_index)
    # # print("Street index:", env.state.street_index)
    # print("small_blind", env.state.actor_index)
