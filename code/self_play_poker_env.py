"""
self_play_poker_env.py
"""

import glob
import os
import random
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pokerkit import Automation, NoLimitTexasHoldem
from sb3_contrib.common.wrappers import ActionMasker

from bot import PokerBot

RANKS = "23456789TJQKA"
SUITS = "cdhs"
RANK_TO_IDX: dict[str, int] = {r: i + 1 for i, r in enumerate(RANKS)}
SUIT_TO_IDX: dict[str, int] = {s: j for j, s in enumerate(SUITS)}


def encode_card(card: str) -> np.ndarray:
    vec = np.zeros(5, dtype=np.float32)
    rank, suit = card[0], card[1]
    vec[0] = RANK_TO_IDX[rank] / 13.0
    vec[1 + SUIT_TO_IDX[suit]] = 1.0
    return vec


def cards_to_vector(cards: list[str]) -> np.ndarray:
    vectors = [encode_card(c) for c in cards]
    if not vectors:
        return np.zeros(5, dtype=np.float32)
    return np.concatenate(vectors)


def make_masked_env(env: "SP_PokerEnv") -> ActionMasker:
    """Wrap an SP_PokerEnv with ActionMasker so MaskablePPO can use legal masks."""
    return ActionMasker(env, lambda e: np.array(e.action_masks(), dtype=bool))


class SP_PokerEnv(gym.Env):
    metadata = {"render_modes": []}

    FOLD = 0
    CHECK = 1
    CALL = 2
    RAISE_QUARTER = 3
    RAISE_HALF = 4
    RAISE_POT = 5

    def __init__(
        self,
        buy_in: float = 100.0,
        sb: float = 5.0,
        bb: float = 10.0,
        snapshot_dir: str = "./snapshots",
        opponent_refresh_freq: int = 200,
        snapshot_pool_size: int = 10,
    ):
        super().__init__()
        self.buy_in = buy_in
        self.sb = sb
        self.bb = bb
        self.snapshot_dir = snapshot_dir
        self.opponent_refresh_freq = opponent_refresh_freq
        self.snapshot_pool_size = snapshot_pool_size

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(54,), dtype=np.float32)
        self.action_space = spaces.Discrete(PokerBot.NUM_ACTIONS)

        self.agent = PokerBot("Agent")
        self.opp   = PokerBot("Opponent")

        self.state: Any = None
        self.done = True
        self.last_actions = [-1, -1]

        self._opponent_model = None
        self._episode_count  = 0
        self._load_opponent()

    # ── action masking ───────────────────────────────────────────────
    def action_masks(self) -> list[bool]:
        """
        Called by MaskablePPO every step. Returns a boolean mask over the
        action space — True = legal, False = illegal. Illegal actions get
        their logits set to -inf before sampling so the agent never picks them.
        """
        return [bool(v) for v in self._legal_mask()]

    # ── opponent management ──────────────────────────────────────────
    def _available_snapshots(self) -> list[str]:
        pattern = os.path.join(self.snapshot_dir, "snapshot_*.zip")
        snaps = sorted(glob.glob(pattern))
        return snaps[-self.snapshot_pool_size:]

    def _load_opponent(self):
        snaps = self._available_snapshots()
        if not snaps:
            self._opponent_model = None
            return
        chosen = random.choice(snaps)
        try:
            from sb3_contrib import MaskablePPO
            self._opponent_model = MaskablePPO.load(chosen)
        except Exception as e:
            print(f"[SP_PokerEnv] Failed to load snapshot {chosen}: {e}")
            self._opponent_model = None

    def _opponent_action(self, obs: np.ndarray) -> int:
        """Get action from opponent model or fall back to random."""
        mask = self._legal_mask()
        if self._opponent_model is not None:
            try:
                arr = np.asarray(obs, dtype=np.float32)
                # Pass the mask so the opponent also samples legally
                action, _ = self._opponent_model.predict(
                    arr,
                    deterministic=False,
                    action_masks=np.array(mask, dtype=bool),
                )
                return int(action)
            except Exception:
                pass
        legal = [i for i, v in enumerate(mask) if v]
        return random.choice(legal) if legal else self.CHECK

    # ── public API ───────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._episode_count += 1
        if self._episode_count % self.opponent_refresh_freq == 0:
            self._load_opponent()
        self._new_hand()
        obs  = self._build_obs()
        info = {"legal_mask": self._legal_mask()}
        return obs, info

    def step(self, action: int):
        # Save legal mask before executing so fold-when-check-was-free penalty works
        self._legal_mask_at_time_of_action = self._legal_mask()

        # Agent is always player 0. Run opponent first if it's their turn
        # (e.g. heads-up preflop: SB=player0 acts first, so opponent doesn't
        # go first; but postflop BB=player1 acts first).
        self._maybe_run_opponent()

        # If opponent finished the hand (e.g. they folded), skip agent action
        if not self._hand_over():
            success = self._execute_action(action, player=0)
            if not success and action == self.FOLD:
                try:
                    self.state.fold()
                except Exception:
                    try:
                        self.state.check_or_call()
                    except Exception:
                        pass

            self._deal_board_if_needed()

            # Let opponent respond if needed
            if not self._hand_over() and self.state.actor_index == 1:
                self._maybe_run_opponent()
                self._deal_board_if_needed()

        reward = 0.0
        done   = False

        # Small penalty for folding when check was free
        mask_before = self._legal_mask_at_time_of_action
        if action == self.FOLD and mask_before[self.CHECK] == 1:
            reward -= 0.05

        if self._hand_over():
            print(f"Hand over. Agent stack: {self.state.stacks[0]}, Opponent stack: {self.state.stacks[1]}")
            reward += self._resolve_hand()
            done        = True
            self.done   = True
            obs  = np.zeros(54, dtype=np.float32)
            info = {"legal_mask": [0, 1, 0, 0, 0, 0]}
        else:
            obs  = self._build_obs()
            info = {"legal_mask": self._legal_mask()}

        return obs, reward, done, False, info

    # ── internal: hand lifecycle ─────────────────────────────────────
    def _new_hand(self):
        self.agent.cash_val = self.buy_in
        self.opp.cash_val   = self.buy_in

        self.state = NoLimitTexasHoldem.create_state(
            (
                Automation.ANTE_POSTING,
                Automation.BET_COLLECTION,
                Automation.BLIND_OR_STRADDLE_POSTING,
                Automation.CARD_BURNING,
                Automation.HAND_KILLING,
                Automation.CHIPS_PUSHING,
                Automation.CHIPS_PULLING,
            ),
            True, 0, (self.sb, self.bb), self.bb,
            (self.buy_in, self.buy_in), 2,
        )

        self.state.deal_hole(self._random_hand())
        self.state.deal_hole(self._random_hand())
        self.done         = False
        self.last_actions = [-1, -1]

    # ── internal: opponent logic ─────────────────────────────────────
    def _maybe_run_opponent(self):
        """Let opponent act for as long as it's their turn."""
        current_round = self._betting_round()
        max_actions   = 10
        actions_taken = 0

        while not self._hand_over() and self.state.actor_index == 1:
            if actions_taken >= max_actions:
                break

            obs            = self._build_obs(player=1)
            opp_action     = self._opponent_action(obs)
            prev_actor_idx = self.state.actor_index

            self.last_actions[1] = opp_action
            success = self._execute_action(opp_action, player=1)
            self._deal_board_if_needed()
            actions_taken += 1

            if not success and self.state.actor_index == prev_actor_idx and not self._hand_over():
                break

            if self._betting_round() != current_round:
                break

    # ── internal: action execution ───────────────────────────────────
    def _execute_action(self, action: int, player: int) -> bool:
        try:
            self.last_actions[player] = action

            if action == self.FOLD:
                self.state.fold()

            elif action in (self.CHECK, self.CALL):
                self.state.check_or_call()

            elif action in (self.RAISE_QUARTER, self.RAISE_HALF, self.RAISE_POT):
                pot   = float(self.state.total_pot_amount) if hasattr(self.state, 'total_pot_amount') else self._current_pot()
                stack = float(self.state.stacks[player])
                multiplier = {
                    self.RAISE_QUARTER: 0.25,
                    self.RAISE_HALF:    0.50,
                    self.RAISE_POT:     1.00,
                }[action]

                raise_amount = pot * multiplier
                current_bet  = float(self.state.bets[player]) if player < len(self.state.bets) else 0.0
                target       = current_bet + raise_amount
                target       = min(target, stack + current_bet)
                min_raise    = self.state.min_completion_betting_or_raising_to_amount

                if min_raise is not None:
                    target = max(target, float(min_raise))

                self.state.complete_bet_or_raise_to(target)

            self._deal_board_if_needed()
            return True

        except Exception:
            if action != self.FOLD:
                try:
                    self.state.check_or_call()
                    return True
                except Exception:
                    pass
            return False

    # ── internal: board dealing ──────────────────────────────────────
    def _deal_board_if_needed(self):
        if self._hand_over():
            return
        max_deals  = 4
        deals_done = 0
        while hasattr(self.state, 'can_deal_board') and self.state.can_deal_board():
            if deals_done >= max_deals:
                break
            try:
                cards_needed = self.state.board_dealing_count
                self.state.deal_board(self._random_cards(cards_needed))
                deals_done += 1
            except Exception:
                break

    # ── internal: observation ────────────────────────────────────────
    def _hand_strength(self, player):
        if not self.state.board_cards:
            return 0.0
        try:
            hand = self.state.get_hand(player, 0, 0)
        except Exception:
            return 0.0
        if hand is None:
            return 0.0
        hand_str = str(hand).split('(')[0].strip().lower()
        rank_values = {
            'high card': 0, 'one pair': 1, 'two pair': 2,
            'three of a kind': 3, 'straight': 4, 'flush': 5,
            'full house': 6, 'four of a kind': 7,
            'straight flush': 8, 'royal flush': 9,
        }
        return rank_values.get(hand_str, 0.0) / 9.0

    def _one_hot_betting_round(self) -> np.ndarray:
        vec = np.zeros(4, dtype=np.float32)
        vec[self._betting_round()] = 1.0
        return vec

    def _one_hot_opponent_last_action(self, player_for_obs: int = 0) -> np.ndarray:
        vec = np.zeros(6, dtype=np.float32)
        opp_action = self.last_actions[1 - player_for_obs]
        if 0 <= opp_action < 6:
            vec[opp_action] = 1.0
        return vec

    def _build_obs(self, player: int = 0) -> np.ndarray:
        obs = np.zeros(54, dtype=np.float32)

        hole_cards, community_cards = [], []
        if self.state.hole_cards and self.state.hole_cards[player]:
            hole_cards += [c.rank + c.suit for c in self.state.hole_cards[player]]
        if self.state.board_cards:
            for part in self.state.board_cards:
                if not part:
                    continue
                if isinstance(part, (list, tuple)):
                    for c in part:
                        community_cards.append(c.rank + c.suit)
                else:
                    community_cards.append(part.rank + part.suit)

        obs[0:10] = cards_to_vector(hole_cards)
        board_vec = cards_to_vector(community_cards)
        obs[10:10 + len(board_vec)] = board_vec

        obs[35] = 0.0 if player == 0 else 1.0
        obs[36] = float(self.state.stacks[player])     / self.buy_in
        obs[37] = float(self.state.stacks[1 - player]) / self.buy_in
        obs[38] = self._current_pot() / (2 * self.buy_in)
        obs[39] = self._hand_strength(player)
        obs[40:44] = self._one_hot_betting_round()
        obs[44:50] = self._one_hot_opponent_last_action(player)

        return obs

    # ── internal: legal action mask ──────────────────────────────────
    def _legal_mask(self) -> list[int]:
        mask = [0, 0, 0, 0, 0, 0]
        if self._hand_over():
            mask[self.CHECK] = 1
            return mask

        mask[self.FOLD] = 1
        try:
            if self.state.can_check_or_call():
                bets    = list(self.state.bets)
                my_bet  = float(bets[0]) if len(bets) > 0 else 0.0
                opp_bet = float(bets[1]) if len(bets) > 1 else 0.0
                if opp_bet > my_bet:
                    mask[self.CALL] = 1
                else:
                    mask[self.CHECK] = 1
            if self.state.can_complete_bet_or_raise_to():
                mask[self.RAISE_QUARTER] = 1
                mask[self.RAISE_HALF]    = 1
                mask[self.RAISE_POT]     = 1
        except Exception:
            mask[self.CHECK] = 1

        return mask

    # ── internal: hand resolution ────────────────────────────────────
    def _resolve_hand(self) -> float:
        agent_final = float(self.state.stacks[0])
        net = agent_final - float(self.buy_in)
        if net > 0:
            self.agent.record_win(net / float(self.buy_in))
        elif net < 0:
            self.agent.record_loss(-net / float(self.buy_in))
        return net / float(self.buy_in)

    # ── internal: utility ────────────────────────────────────────────
    def _hand_over(self) -> bool:
        if self.state.actor_index is not None:
            return False
        if hasattr(self.state, 'can_deal_board') and self.state.can_deal_board():
            return False
        return True

    def _current_pot(self) -> float:
        return float(self.state.total_pot_amount) if hasattr(self.state, 'total_pot_amount') else 0.0

    def _betting_round(self) -> int:
        n = len(self.state.board_cards) if self.state.board_cards else 0
        if n == 0:   return 0
        elif n == 3: return 1
        elif n == 4: return 2
        else:        return 3

    def _random_hand(self) -> str:
        dealable = list(self.state.get_dealable_cards())
        cards = random.sample(dealable, 2)
        return "".join(c.rank + c.suit for c in cards)

    def _random_cards(self, n: int) -> str:
        dealable = list(self.state.get_dealable_cards())
        cards = random.sample(dealable, n)
        return "".join(c.rank + c.suit for c in cards)