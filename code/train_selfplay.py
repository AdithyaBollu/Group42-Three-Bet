"""
train_selfplay.py
-----------------
Train a single MaskablePPO agent with snapshot-based self-play.

Usage:
    python train_selfplay.py --steps 1000000
    python train_selfplay.py --steps 500000 --resume snapshots/final_500000.zip
"""

import argparse
import os
import numpy as np
from collections import deque

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback

from self_play_poker_env import SP_PokerEnv, make_masked_env

# ── hyper-parameters ─────────────────────────────────────────────────
DEFAULTS = dict(
    total_timesteps = 1_000_000,
    n_steps         = 2048,
    batch_size      = 256,
    n_epochs        = 10,
    learning_rate   = 3e-4,
    gamma           = 0.95,
    gae_lambda      = 0.9,
    clip_range      = 0.2,
    ent_coef        = 0.01,
    verbose         = 1,
)

SNAPSHOT_DIR = "./masked_snaps"
SNAPSHOT_FREQ = 50_000


# ── env factory ──────────────────────────────────────────────────────
def make_train_env():
    env = SP_PokerEnv(
        buy_in=75.0,
        sb=5.0,
        bb=10.0,
        snapshot_dir=SNAPSHOT_DIR,
        opponent_refresh_freq=300,
        snapshot_pool_size=10,
    )
    return make_masked_env(env)


def make_eval_env():
    env = SP_PokerEnv(
        buy_in=75.0,
        sb=5.0,
        bb=10.0,
        snapshot_dir=SNAPSHOT_DIR,
        opponent_refresh_freq=100,
        snapshot_pool_size=10,
    )
    return make_masked_env(env)


# ── callbacks ────────────────────────────────────────────────────────
class WinLossCallback(BaseCallback):
    """Track wins/losses/pushes and log rolling stats to TensorBoard."""

    def __init__(self, print_every: int = 100, rolling_window: int = 200, verbose: int = 1):
        super().__init__(verbose)
        self.print_every = print_every
        self.rolling_window = rolling_window

        self.episode_count = 0
        self.wins = 0
        self.losses = 0
        self.pushes = 0
        self.total_reward = 0.0

        self.episode_rewards = deque(maxlen=self.rolling_window)
        self.episode_lengths = deque(maxlen=self.rolling_window)

    def _on_step(self) -> bool:
        infos   = self.locals.get("infos", []) or []
        rewards = self.locals.get("rewards", []) or []
        dones   = self.locals.get("dones", []) or []

        for i, info in enumerate(infos):
            done_flag = isinstance(dones, (list, tuple, np.ndarray)) and i < len(dones) and dones[i]

            if "terminal_observation" in info or info.get("TimeLimit.truncated", False) or done_flag:
                self.episode_count += 1
                reward = float(rewards[i]) if i < len(rewards) else 0.0
                self.total_reward += reward
                self.episode_rewards.append(reward)

                ep_len = 0
                if isinstance(info.get("episode"), dict):
                    ep_len = int(info["episode"].get("l", 0))
                elif "length" in info:
                    ep_len = int(info.get("length", 0))
                self.episode_lengths.append(ep_len)

                if reward > 0:
                    self.wins += 1
                elif reward < 0:
                    self.losses += 1
                else:
                    self.pushes += 1

                rolling_win_rate   = sum(1 for r in self.episode_rewards if r > 0) / len(self.episode_rewards) * 100 if self.episode_rewards else 0.0
                rolling_avg_reward = sum(self.episode_rewards) / len(self.episode_rewards) if self.episode_rewards else 0.0
                avg_length         = sum(self.episode_lengths) / len(self.episode_lengths) if self.episode_lengths else 0.0

                if self.episode_count % self.print_every == 0:
                    overall_win_rate   = (self.wins / self.episode_count) * 100
                    overall_avg_reward = self.total_reward / self.episode_count
                    print("\n" + "=" * 60)
                    print(f"Episodes: {self.episode_count} | W:{self.wins} L:{self.losses} P:{self.pushes}")
                    print(f"Overall  Win Rate: {overall_win_rate:.2f}%  |  Avg Reward: {overall_avg_reward:.4f}")
                    print(f"Rolling({self.rolling_window}) Win Rate: {rolling_win_rate:.2f}%  |  Avg Reward: {rolling_avg_reward:.4f}")
                    print(f"Avg Episode Length (rolling): {avg_length:.1f}")
                    print("=" * 60)

                try:
                    self.logger.record("stats/episodes",           self.episode_count)
                    self.logger.record("stats/overall_win_rate",   overall_win_rate)
                    self.logger.record("stats/rolling_win_rate",   rolling_win_rate)
                    self.logger.record("stats/rolling_avg_reward", rolling_avg_reward)
                    self.logger.record("stats/rolling_avg_length", avg_length)
                    self.logger.record("stats/wins",               self.wins)
                    self.logger.record("stats/losses",             self.losses)
                    self.logger.record("stats/pushes",             self.pushes)
                except Exception:
                    pass

        return True


class SnapshotCallback(BaseCallback):
    """Save model snapshots every `freq` timesteps."""

    def __init__(self, snapshot_dir: str = SNAPSHOT_DIR, freq: int = SNAPSHOT_FREQ, verbose: int = 1):
        super().__init__(verbose)
        self.snapshot_dir = snapshot_dir
        self.freq = int(freq)
        self.last_saved = 0
        os.makedirs(self.snapshot_dir, exist_ok=True)

    def _on_step(self) -> bool:
        try:
            if (self.num_timesteps - self.last_saved) >= self.freq:
                path = os.path.join(self.snapshot_dir, f"snapshot_{int(self.num_timesteps)}")
                self.model.save(path)
                self.last_saved = self.num_timesteps
                if self.verbose:
                    print(f"[Snapshot] Saved: {path}.zip")
        except Exception as e:
            if self.verbose:
                print(f"[Snapshot] Save failed: {e}")
        return True


# ── self-play evaluation ─────────────────────────────────────────────
def run_selfplay_with_model(model: MaskablePPO, games: int = 200):
    """Run evaluation where the trained model plays both seats."""
    env = SP_PokerEnv(buy_in=75.0, sb=5.0, bb=10.0)
    stats = {"p0": 0, "p1": 0, "push": 0}

    def policy(obs, mask):
        arr = np.asarray(obs, dtype=np.float32)
        action, _ = model.predict(arr, deterministic=True, action_masks=np.array(mask, dtype=bool))
        return int(action)

    for _ in range(games):
        env._new_hand()
        env.last_actions = [-1, -1]
        while True:
            if env._hand_over():
                r = env._resolve_hand()
                if r > 0:   stats["p0"] += 1
                elif r < 0: stats["p1"] += 1
                else:       stats["push"] += 1
                break

            actor = env.state.actor_index
            if actor is None:
                break
            obs    = env._build_obs(player=actor)
            mask   = env._legal_mask()
            action = policy(obs, mask)
            env._execute_action(action, player=actor)
            env._deal_board_if_needed()


# ── training entry point ─────────────────────────────────────────────
def train(
    total_timesteps: int = DEFAULTS["total_timesteps"],
    eval_games: int = 200,
    snapshot_freq: int = SNAPSHOT_FREQ,
    resume_path: str | None = None,
):
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    vec_env = DummyVecEnv([make_train_env])

    if resume_path:
        print(f"Resuming from: {resume_path}")
        model = MaskablePPO.load(resume_path, env=vec_env)
    else:
        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[256, 256, 256])),
            n_steps       = DEFAULTS["n_steps"],
            batch_size    = DEFAULTS["batch_size"],
            n_epochs      = DEFAULTS["n_epochs"],
            learning_rate = DEFAULTS["learning_rate"],
            gamma         = DEFAULTS["gamma"],
            gae_lambda    = DEFAULTS["gae_lambda"],
            clip_range    = DEFAULTS["clip_range"],
            ent_coef      = DEFAULTS["ent_coef"],
            verbose       = DEFAULTS["verbose"],
            tensorboard_log="./tensorboard_logs/",
        )

    eval_env = DummyVecEnv([make_eval_env])
    eval_cb  = EvalCallback(
        eval_env,
        eval_freq=5_000,
        n_eval_episodes=200,
        verbose=1,
    )

    snapshot_cb = SnapshotCallback(snapshot_dir=SNAPSHOT_DIR, freq=snapshot_freq, verbose=1)

    print("Starting MaskablePPO snapshot self-play training…")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_cb, snapshot_cb],
    )

    model.save("poker_ppo_selfplay_model")
    print("Model saved → poker_ppo_selfplay_model.zip")

    final_path = os.path.join(SNAPSHOT_DIR, f"final_{int(total_timesteps)}")
    model.save(final_path)
    print(f"Final snapshot saved → {final_path}.zip")

    run_selfplay_with_model(model, games=eval_games)
    return model


# ── CLI ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MaskablePPO with snapshot self-play.")
    parser.add_argument("--steps",         type=int,   default=DEFAULTS["total_timesteps"])
    parser.add_argument("--eval-games",    type=int,   default=200)
    parser.add_argument("--snapshot-freq", type=int,   default=SNAPSHOT_FREQ)
    parser.add_argument("--resume",        type=str,   default=None)
    args = parser.parse_args()

    train(
        total_timesteps=args.steps,
        eval_games=args.eval_games,
        snapshot_freq=args.snapshot_freq,
        resume_path=args.resume,
    )