import glob
import os
import random
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pokerkit import Automation, NoLimitTexasHoldem

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


class FixedPokerEnvironment(gym.Env):
    metadata = {"render_modes": []}

    FOLD = 0
    CHECK = 1
    CALL = 2
    RAISE_QUARTER = 3
    RAISE_HALF = 4
    RAISE_POT = 5

    # Number of past actions encoded in each observation.
    # Each slot = 7 dims: [who_acted | one-hot action (6)]
    ACTION_HISTORY_LEN = 6
    OBS_DIM = 43 + ACTION_HISTORY_LEN * 7  # 85

    def __init__(
        self,
        buy_in: float = 100.0,
        sb: float = 5.0,
        bb: float = 10.0,
        snapshot_dir: str = "./snapshots",
        opponent_refresh_freq: int = 500,
        snapshot_pool_size: int = 10,
        player_type: str = "A",
    ):
        super().__init__()

        self.buy_in = buy_in
        self.sb = sb
        self.bb = bb
        self.snapshot_dir = snapshot_dir
        self.opponent_refresh_freq = opponent_refresh_freq
        self.snapshot_pool_size = snapshot_pool_size
        self.player_type = player_type

        self.player_seat = 0  # toggled each reset

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(PokerBot.NUM_ACTIONS)

        self.agent = PokerBot("Agent")
        self.opp = PokerBot("Opponent")

        self.state: Any = None
        self.done = True
        self.last_actions   = [-1, -1]
        # Chronological list of (actor_seat, action_int) for the current hand.
        # Sliding window of the last ACTION_HISTORY_LEN entries goes into obs.
        self.action_history: list[tuple[int, int]] = []

        self._opponent_model = None
        self._episode_count = 0
        self._load_opponent()

    # ------------------------------------------------------------
    # Card helpers
    # ------------------------------------------------------------

    def _random_hand(self) -> str:
        dealable = list(self.state.get_dealable_cards())
        cards = random.sample(dealable, 2)
        return "".join(c.rank + c.suit for c in cards)

    # ------------------------------------------------------------
    # Opponent loading
    # ------------------------------------------------------------

    def _available_opponent_snapshots(self) -> list[str]:
        pattern = os.path.join(self.snapshot_dir + self.player_type, "snapshot_*.zip")
        snaps = glob.glob(pattern)
        # Sort numerically by the timestep number embedded in the filename
        # e.g. snapshot_50000.zip → 50000, snapshot_100000.zip → 100000
        def _ts(p: str) -> int:
            try:
                return int(os.path.basename(p).replace("snapshot_", "").replace(".zip", ""))
            except ValueError:
                return 0
        snaps = sorted(snaps, key=_ts)
        return snaps[-self.snapshot_pool_size :]

    def _load_opponent(self):
        snaps = self._available_opponent_snapshots()
        if not snaps:
            self._opponent_model = None
            return

        chosen = snaps[-1]  # always use the latest snapshot
        try:
            from sb3_contrib import MaskablePPO
            self._opponent_model = MaskablePPO.load(chosen)
            print(f"  [opponent] loaded snapshot: {os.path.basename(chosen)}")
        except Exception as e:
            print(f"  [opponent] FAILED to load {chosen}: {e}")
            self._opponent_model = None

    # ------------------------------------------------------------
    # Legal mask
    # ------------------------------------------------------------

    def _legal_mask(self, player_seat) -> list[int]:
        mask = [0, 0, 0, 0, 0, 0]

        if self._hand_over():
            mask[self.CHECK] = 1
            return mask

        mask[self.FOLD] = 1

        try:
            if self.state.can_check_or_call():
                bets = list(self.state.bets)
                my_bet = float(bets[player_seat]) if len(bets) > 0 else 0.0
                opp_bet = float(bets[1 - player_seat]) if len(bets) > 1 else 0.0

                if opp_bet > my_bet:
                    mask[self.CALL] = 1
                else:
                    # Can check for free — folding is never correct, make it illegal
                    mask[self.CHECK] = 1
                    mask[self.FOLD]  = 0

            if self.state.can_complete_bet_or_raise_to():
                mask[self.RAISE_QUARTER] = 1
                mask[self.RAISE_HALF] = 1
                mask[self.RAISE_POT] = 1

        except Exception:
            mask[self.CHECK] = 1
            mask[self.FOLD]  = 0

        return mask

    # ------------------------------------------------------------
    # Opponent action
    # ------------------------------------------------------------

    def _opponent_action(self, obs: np.ndarray, player_seat: int) -> int:
        mask = self._legal_mask(player_seat)
        legal = [i for i, v in enumerate(mask) if v]

        if self._opponent_model is not None:
            try:
                arr = np.asarray(obs, dtype=np.float32).reshape(1, -1)
                action, _ = self._opponent_model.predict(
                    arr,
                    action_masks=np.array(mask, dtype=np.float32),
                    deterministic=False,
                )
                return int(action)
            except Exception:
                return random.choice(legal) if legal else self.CHECK

        return random.choice(legal) if legal else self.CHECK

    # ------------------------------------------------------------
    # New hand (FIXED dealer rotation + hole assignment)
    # ------------------------------------------------------------

    def _new_hand(self):
        self.agent.cash_val = self.buy_in
        self.opp.cash_val = self.buy_in

        dealer_position = self.player_seat

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
            True,
            0,               # antes = 0 (no ante); dealer_position does not map here
            (self.sb, self.bb),
            self.bb,
            (self.buy_in, self.buy_in),
            2,
        )

        sb_hand = self._random_hand()
        self.state.deal_hole(sb_hand)

        bb_hand = self._random_hand()
        self.state.deal_hole(bb_hand)

        # FIXED hole assignment
        if self.player_seat == 0:
            self.players_cards = sb_hand
            self.opponents_cards = bb_hand
        else:
            self.players_cards = bb_hand
            self.opponents_cards = sb_hand

        self.done           = False
        self.last_actions   = [-1, -1]
        self.action_history = []

    # ------------------------------------------------------------
    # Observation (FIXED signature + zeroing board region)
    # ------------------------------------------------------------

    def _build_obs(self, player_position: int) -> np.ndarray:
        obs = np.zeros(self.OBS_DIM, dtype=np.float32)

        hole_cards = []
        community_cards = []

        if self.state.hole_cards and self.state.hole_cards[player_position]:
            hole_cards += [
                c.rank + c.suit
                for c in self.state.hole_cards[player_position]
            ]

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

        # FIXED zeroing
        obs[10:35] = 0.0
        board_vec = cards_to_vector(community_cards)
        obs[10 : 10 + len(board_vec)] = board_vec

        obs[35] = 0.0 if player_position == 0 else 1.0
        obs[36] = float(self.state.stacks[player_position]) / self.buy_in
        obs[37] = float(self.state.stacks[1 - player_position]) / self.buy_in
        obs[38] = self.state.total_pot_amount / (2 * self.buy_in)

        # Betting round via board card count (0=preflop,1=flop,2=turn,3=river)
        n_board = len(self.state.board_cards) if self.state.board_cards else 0
        if   n_board == 0: street_idx = 0
        elif n_board == 3: street_idx = 1
        elif n_board == 4: street_idx = 2
        else:              street_idx = 3
        street_vec = np.zeros(4, dtype=np.float32)
        street_vec[street_idx] = 1.0
        obs[39:43] = street_vec

        # [43 : 43 + ACTION_HISTORY_LEN*7] — action history (oldest → newest).
        # Each 7-dim slot: [who_acted, fold, check, call, raise_q, raise_h, raise_p]
        #   who_acted: 0.0 = self (player_position), 1.0 = opponent.
        # Unused slots stay all-zero ("no action yet").
        history = self.action_history[-self.ACTION_HISTORY_LEN:]
        for i, (actor_seat, act) in enumerate(history):
            slot = 43 + i * 7
            obs[slot] = 0.0 if actor_seat == player_position else 1.0
            if 0 <= act < 6:
                obs[slot + 1 + act] = 1.0

        return obs

    # ------------------------------------------------------------
    # Turn sequencing (FIXED)
    # ------------------------------------------------------------

    def _maybe_run_opponent(self):
        opponent_seat = 1 - self.player_seat

        while (
            not self._hand_over()
            and self.state.actor_index == opponent_seat
        ):
            obs = self._build_obs(opponent_seat)
            action = self._opponent_action(obs, opponent_seat)
            self._execute_action(action, player=opponent_seat)
            self.last_actions[opponent_seat] = action
            self.action_history.append((opponent_seat, action))

    # ------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self._episode_count += 1

        if self._episode_count % self.opponent_refresh_freq == 0:
            self._load_opponent()

        # FIXED blind alternation
        self.player_seat = 1 - self.player_seat

        # Loop until the agent actually has a decision to make.
        # The opponent may legally fold pre-flop (or otherwise end the hand)
        # before the agent gets to act — retry with the SAME seat so alternation
        # is never disrupted by retries.
        while True:
            self._new_hand()
            self._maybe_run_opponent()  # let opponent act first if needed
            if not self._hand_over():
                break
            # Hand ended before agent could act — try again, same seat

        obs  = self._build_obs(self.player_seat)
        info = {"action_masks": self._legal_mask(self.player_seat)}
        return obs, info

    def step(self, action: int):
        # Safety: if something caused the opponent to still need to act
        # (e.g. a board-deal edge case), resolve that before the agent moves.
        self._maybe_run_opponent()

        # If the hand somehow ended before agent acts, treat as a no-op terminal.
        if self._hand_over():
            net = float(self.state.stacks[self.player_seat]) - float(self.buy_in)
            reward = net / float(self.buy_in)
            obs  = np.zeros(self.OBS_DIM, dtype=np.float32)
            info = {"action_masks": [0, 1, 0, 0, 0, 0]}
            return obs, reward, True, False, info

        if not self._hand_over():
            self._execute_action(action, player=self.player_seat)
            self.action_history.append((self.player_seat, action))

        # Let opponent fully respond
        self._maybe_run_opponent()

        reward = 0.0
        done = False

        if self._hand_over():
            net = float(self.state.stacks[self.player_seat]) - float(self.buy_in)
            reward += net / float(self.buy_in)
            done = True
            obs = np.zeros(self.OBS_DIM, dtype=np.float32)
            info = {"action_masks": [0, 1, 0, 0, 0, 0]}
        else:
            obs = self._build_obs(self.player_seat)
            info = {"action_masks": self._legal_mask(self.player_seat)}

        return obs, reward, done, False, info

    # ------------------------------------------------------------
    # Hand state helpers
    # ------------------------------------------------------------

    def _hand_over(self) -> bool:
        """True when PokerKit considers the hand finished."""
        if self.state.actor_index is not None:
            return False
        if hasattr(self.state, "can_deal_board") and self.state.can_deal_board():
            return False
        return True

    def _current_pot(self) -> float:
        try:
            return float(self.state.total_pot_amount)
        except Exception:
            return 0.0

    def _random_cards(self, n: int) -> str:
        dealable = list(self.state.get_dealable_cards())
        cards = random.sample(dealable, n)
        return "".join(c.rank + c.suit for c in cards)

    def _deal_board_if_needed(self):
        """Deal community cards whenever PokerKit is waiting for them."""
        if self._hand_over():
            return
        max_deals = 4
        deals_done = 0
        while hasattr(self.state, "can_deal_board") and self.state.can_deal_board():
            if deals_done >= max_deals:
                break
            try:
                n = self.state.board_dealing_count
                self.state.deal_board(self._random_cards(n))
                deals_done += 1
            except Exception:
                break

    def _execute_action(self, action: int, player: int) -> bool:
        """Translate a discrete action into a PokerKit call. Returns True on success."""
        try:
            if action == self.FOLD:
                self.state.fold()

            elif action in (self.CHECK, self.CALL):
                self.state.check_or_call()

            elif action in (self.RAISE_QUARTER, self.RAISE_HALF, self.RAISE_POT):
                pot        = self._current_pot()
                stack      = float(self.state.stacks[player])
                multiplier = {
                    self.RAISE_QUARTER: 0.25,
                    self.RAISE_HALF:    0.50,
                    self.RAISE_POT:     1.00,
                }[action]
                current_bet = (
                    float(self.state.bets[player])
                    if player < len(self.state.bets) else 0.0
                )
                target    = current_bet + pot * multiplier
                target    = min(target, stack + current_bet)  # all-in cap
                min_raise = self.state.min_completion_betting_or_raising_to_amount
                if min_raise is not None:
                    target = max(target, float(min_raise))
                self.state.complete_bet_or_raise_to(target)

            self._deal_board_if_needed()
            return True

        except Exception:
            # Illegal action — fall back to check/call rather than crashing
            if action != self.FOLD:
                try:
                    self.state.check_or_call()
                    self._deal_board_if_needed()
                    return True
                except Exception:
                    pass
            return False