"""
train2.py
---------
Trains a MaskablePPO agent on FixedPokerEnvironment (poker_env_3).
Action masking eliminates illegal moves from the policy logits before
sampling, so the agent never wastes gradient signal on impossible actions.

Usage
-----
    python train2.py                   # 1 000 000 timesteps (default)
    python train2.py --steps 2000000   # longer run
    python train2.py --steps 500000 --snapshot-freq 50000

Snapshots
---------
A snapshot is saved every --snapshot-freq steps so the self-play opponent
pool stays fresh. Final model is saved as poker_masked_ppo_final.zip.
"""

import argparse
import os

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from poker_env_3 import FixedPokerEnvironment
from stats import _empty, _record, get_sorted_snapshots, play_one_hand


# ── helpers ──────────────────────────────────────────────────────────

def mask_fn(env) -> np.ndarray:
    """Called by ActionMasker every step to get the legal-action boolean mask."""
    # Unwrap Monitor (or any other wrapper) to reach FixedPokerEnvironment
    inner = env
    while hasattr(inner, "env"):
        inner = inner.env
    return np.array(inner._legal_mask(inner.player_seat), dtype=bool)


def make_env(snapshot_dir: str = "./masked_snaps", player_type: str = "A"):
    """Factory for DummyVecEnv — wraps env with ActionMasker."""
    def _init():
        env = FixedPokerEnvironment(
            buy_in=100.0,
            sb=5.0,
            bb=10.0,
            snapshot_dir=snapshot_dir,
            opponent_refresh_freq=15_000,
            snapshot_pool_size=10,
            player_type=player_type,
        )
        env = Monitor(env)   # populates info["episode"] at episode end
        return ActionMasker(env, mask_fn)
    return _init


# ── callbacks ─────────────────────────────────────────────────────────

class SnapshotCallback(BaseCallback):
    """Saves a model snapshot every `save_freq` steps for the opponent pool."""

    def __init__(self, save_freq: int, snapshot_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq    = save_freq
        self.snapshot_dir = snapshot_dir
        os.makedirs(snapshot_dir + "A", exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            path = os.path.join(
                self.snapshot_dir + "A",
                f"snapshot_{self.num_timesteps}"
            )
            self.model.save(path)
            if self.verbose:
                print(f"  [snapshot] saved → {path}.zip")
        return True


class WinLossCallback(BaseCallback):
    """Prints rolling win/loss/push stats every `print_every` episodes."""

    WINDOW = 500  # rolling window size

    def __init__(self, print_every: int = 200, verbose: int = 1):
        super().__init__(verbose)
        self.print_every   = print_every
        self.episode_count = 0
        self.total_wins    = 0
        self.total_losses  = 0
        self.total_pushes  = 0
        self.total_reward  = 0.0
        self._window: list[float] = []  # rolling reward buffer

    def _on_step(self) -> bool:
        for i, info in enumerate(self.locals.get("infos", [])):
            # SB3 DummyVecEnv stores episode summary in info["episode"] at
            # the terminal step: {"r": total_reward, "l": episode_length}
            ep_info = info.get("episode")
            if ep_info is None:
                continue

            reward = float(ep_info["r"])
            self.episode_count += 1
            self.total_reward += reward

            if   reward > 0: 
                self.total_wins   += 1
            elif reward < 0: 
                self.total_losses += 1
            else:            
                self.total_pushes += 1

            # rolling window
            self._window.append(reward)
            if len(self._window) > self.WINDOW:
                self._window.pop(0)

            if self.episode_count % self.print_every == 0:
                n   = len(self._window)
                rw  = sum(1 for r in self._window if r > 0) / n * 100
                avg = sum(self._window) / n
                print(
                    f"\n  {'─'*55}\n"
                    f"  ep {self.episode_count:>6} | "
                    f"W {self.total_wins} L {self.total_losses} | "
                    f"rolling win% {rw:.1f}% | rolling avg {avg:+.3f}  "
                    f"(last {n} hands)\n"
                    f"  {'─'*55}"
                )
        return True


class EvalVsSnapshotCallback(BaseCallback):
    """
    Every `eval_freq` training steps:
      - loads the latest snapshot from snapshot_dir
      - plays `n_eval_hands` hands (alternating seats) against the current model
      - appends a line to `log_file`
      - saves/updates `plot_file` with win% and avg-chip charts
    """

    def __init__(self, eval_freq: int, snapshot_dir: str,
                 n_eval_hands: int = 300,
                 log_file: str  = "eval_log.txt",
                 verbose: int   = 1):
        super().__init__(verbose)
        self.eval_freq    = eval_freq
        self.snapshot_dir = snapshot_dir
        self.n_eval_hands = n_eval_hands
        self.log_file     = log_file

        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"{'steps':>10}  {'win%':>7}  {'BB%':>7}  {'SB%':>7}  "
                    f"{'avg chips':>10}  snapshot\n")
            f.write("-" * 75 + "\n")

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq != 0:
            return True

        snaps = get_sorted_snapshots(self.snapshot_dir)
        if not snaps:
            return True

        latest = snaps[-1]
        try:
            opp = MaskablePPO.load(latest)
        except Exception as e:
            print(f"  [eval] failed to load {latest}: {e}")
            return True

        env = FixedPokerEnvironment(buy_in=100.0, sb=5.0, bb=10.0,
                                    snapshot_dir=self.snapshot_dir)
        env._opponent_model = None

        overall = _empty()
        bb_rec  = _empty()   # agent sat in seat 0 (BB)
        sb_rec  = _empty()   # agent sat in seat 1 (SB)

        for idx in range(self.n_eval_hands):
            seat_a = idx % 2
            net_a, _ = play_one_hand(env, self.model, opp, seat_a)
            _record(overall, net_a)
            if seat_a == 0:
                _record(bb_rec, net_a)
            else:
                _record(sb_rec, net_a)

        n     = self.n_eval_hands
        half  = n // 2
        wp    = overall["wins"] / n    * 100
        bb_wp = bb_rec["wins"]  / half * 100 if half   else 0.0
        sb_wp = sb_rec["wins"]  / (n - half) * 100 if (n - half) else 0.0
        avg   = overall["chip_sum"] / n
        snap_name = os.path.basename(latest)

        # ── TensorBoard ───────────────────────────────────────────
        self.logger.record("eval/win_pct",    wp)
        self.logger.record("eval/bb_win_pct", bb_wp)
        self.logger.record("eval/sb_win_pct", sb_wp)
        self.logger.record("eval/avg_chips",  avg)
        self.logger.dump(self.num_timesteps)

        # ── text log ──────────────────────────────────────────────
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"{self.num_timesteps:>10}  {wp:>6.1f}%  {bb_wp:>6.1f}%  "
                    f"{sb_wp:>6.1f}%  {avg:>+10.2f}  {snap_name}\n")

        if self.verbose:
            print(f"\n  [eval@{self.num_timesteps:,}] "
                  f"win%={wp:.1f}%  BB={bb_wp:.1f}%  SB={sb_wp:.1f}%  "
                  f"avg={avg:+.2f}  ({snap_name})")

        return True

DEFAULTS = dict(
    total_timesteps = 1_000_000,
    n_steps         = 2048,
    batch_size      = 256,
    n_epochs        = 10,
    learning_rate   = 3e-4,
    gamma           = 0.95,
    gae_lambda      = 0.9,
    clip_range      = 0.2,
    ent_coef        = 0.05,
    verbose         = 1,
)


# ── main training function ────────────────────────────────────────────

def train(total_timesteps: int, snapshot_freq: int, snapshot_dir: str, resume: str | None = None):

    os.makedirs(snapshot_dir + "A", exist_ok=True)

    # ── 1. vectorised training env ───────────────────────────────────
    vec_env = DummyVecEnv([make_env(snapshot_dir=snapshot_dir)])

    # ── 2. MaskablePPO agent ─────────────────────────────────────────
    if resume:
        if not os.path.exists(resume) and not os.path.exists(resume + ".zip"):
            raise SystemExit(f"Resume model not found: {resume}")
        print(f"  Resuming from: {resume}")
        model = MaskablePPO.load(
            resume,
            env             = vec_env,
            tensorboard_log = "./tensorboard_logs/",
        )
        # Restore any hyperparams that load() doesn't preserve
        model.ent_coef = DEFAULTS["ent_coef"]
    else:
        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            n_steps         = DEFAULTS["n_steps"],
            batch_size      = DEFAULTS["batch_size"],
            n_epochs        = DEFAULTS["n_epochs"],
            learning_rate   = DEFAULTS["learning_rate"],
            gamma           = DEFAULTS["gamma"],
            gae_lambda      = DEFAULTS["gae_lambda"],
            clip_range      = DEFAULTS["clip_range"],
            ent_coef        = DEFAULTS["ent_coef"],
            verbose         = DEFAULTS["verbose"],
            tensorboard_log = "./tensorboard_logs/",
        )

    # ── 3. callbacks ─────────────────────────────────────────────────
    snap_cb    = SnapshotCallback(snapshot_freq, snapshot_dir, verbose=1)
    winloss_cb = WinLossCallback(print_every=200, verbose=1)
    eval_cb    = EvalVsSnapshotCallback(
        eval_freq    = snapshot_freq,
        snapshot_dir = snapshot_dir,
        n_eval_hands = 300,
        log_file     = "eval_log.txt",
        verbose      = 1,
    )

    # ── 4. train ─────────────────────────────────────────────────────
    print("=" * 60)
    print("  MaskablePPO — poker_env_3 (FixedPokerEnvironment)")
    print(f"  OBS dim     : {FixedPokerEnvironment.OBS_DIM}")
    print(f"  Action hist : {FixedPokerEnvironment.ACTION_HISTORY_LEN} slots")
    print(f"  Timesteps   : {total_timesteps:,}")
    print(f"  Snapshot dir: {snapshot_dir}A/")
    print(f"  Resume from : {resume if resume else 'scratch'}")
    print(f"  Eval log    : eval_log.txt  (also → TensorBoard eval/**)")
    print("=" * 60)

    model.learn(
        total_timesteps      = total_timesteps,
        callback             = [snap_cb, winloss_cb, eval_cb],
        reset_num_timesteps  = resume is None,
    )

    # ── 5. save final model ───────────────────────────────────────────
    final_path = "poker_masked_ppo_final"
    model.save(final_path)
    print(f"\n✓  Final model saved → {final_path}.zip")

    # ── 6. quick post-training eval ───────────────────────────────────
    N_EVAL = 50
    print(f"\n── Post-training evaluation ({N_EVAL} hands) ──")
    raw_env      = FixedPokerEnvironment(buy_in=100.0, sb=5.0, bb=10.0)
    raw_env._opponent_model = None  # pure random opponent — no snapshot loading lag
    eval_env_raw = ActionMasker(raw_env, mask_fn)
    wins = losses = pushes = 0
    total_reward  = 0.0

    for _ in range(N_EVAL):
        obs, info = eval_env_raw.reset()
        done = False
        step_limit = 0
        while not done and step_limit < 50:   # hard cap: prevent infinite loops
            masks  = np.array(info.get("action_masks", mask_fn(raw_env)), dtype=bool)
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            obs, reward, done, truncated, info = eval_env_raw.step(int(action))
            step_limit += 1
        total_reward += reward
        if   reward > 0: wins   += 1
        elif reward < 0: losses += 1
        else:            pushes += 1

    wr = wins / N_EVAL * 100
    print(f"  Wins {wins} | Losses {losses} | Pushes {pushes} | Win% {wr:.1f}%")
    print(f"  Avg reward per hand: {total_reward / N_EVAL:+.3f}")

    return model


# ── CLI ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MaskablePPO poker agent.")
    parser.add_argument(
        "--steps", type=int, default=DEFAULTS["total_timesteps"],
        help="Total training timesteps (default 1 000 000)."
    )
    parser.add_argument(
        "--snapshot-freq", type=int, default=50_000,
        help="Save a snapshot every N timesteps (default 50 000)."
    )
    parser.add_argument(
        "--snapshot-dir", type=str, default="./masked_snaps",
        help="Directory to store opponent snapshots (default ./masked_snaps)."
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to a .zip model to resume training from (e.g. poker_masked_ppo_final)."
    )
    args = parser.parse_args()

    train(
        total_timesteps = args.steps,
        snapshot_freq   = args.snapshot_freq,
        snapshot_dir    = args.snapshot_dir,
        resume          = args.resume,
    )
