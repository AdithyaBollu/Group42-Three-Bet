"""
bot.py
------
Defines the PokerBot class.  Each bot tracks its own bankroll,
cumulative reward signal, and win/loss record.  The reward is what
the PPO agent actually optimises; wins/losses are just for logging.
"""


class PokerBot:
    """Represents one RL-driven poker player."""

    # ── action labels (shared across all bots) ──────────────────────
    ACTIONS = ["fold", "check", "call", "raise_quarter", "raise_half", "raise_pot"]
    NUM_ACTIONS = len(ACTIONS)          # 6

    def __init__(self, name: str, buy_in: float = 75.0):
        self.name: str = name
        self.cash_val: float = buy_in    # current chip stack
        self.reward: float = 0.0         # cumulative RL reward
        self.wins: int = 0
        self.losses: int = 0

    # ── per-hand helpers ─────────────────────────────────────────────
    def record_win(self, amount: float) -> None:
        """Call after the bot wins a hand."""
        self.cash_val += amount
        self.reward += amount            # positive reward signal
        self.wins += 1

    def record_loss(self, amount: float) -> None:
        """Call after the bot loses a hand.  `amount` is positive."""
        self.cash_val -= amount
        self.reward -= amount            # negative reward signal
        self.losses += 1

    # ── bookkeeping ──────────────────────────────────────────────────
    def is_busted(self) -> bool:
        return self.cash_val <= 0.0

    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.0

    def __repr__(self) -> str:
        return (
            f"PokerBot(name={self.name!r}, cash={self.cash_val:.2f}, "
            f"reward={self.reward:.2f}, wins={self.wins}, losses={self.losses})"
        )
