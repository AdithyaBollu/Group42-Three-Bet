"""
train.py
--------
Entry-point.  Creates the PokerEnv, wraps it for Stable-Baselines3,
trains a PPO agent, and plots the learning curve.

Usage
-----
    python train.py                # default: 200 000 timesteps
    python train.py --steps 500000 # longer run

Requirements  (install once)
-----------------------------
    pip install pokerkit gymnasium stable-baselines3 torch matplotlib

How PPO learns here
-------------------
1. The agent plays many hands (rollouts).
2. For each hand it collects:
       (state, action, reward)   ← one tuple per decision point
3. PPO computes an *advantage* for every action:
       advantage = actual_return  –  predicted_return
   This tells the network "was this action better or worse than you
   expected?"
4. The policy (neural net) is updated so that actions with positive
   advantage become more likely, and actions with negative advantage
   become less likely.
5. The "Proximal" part clips the update so the policy doesn't change
   too drastically in one step — this keeps training stable.
6. Repeat thousands of times and the agent gradually discovers
   profitable patterns.
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback

from poker_env import PokerEnv


class WinLossCallback(BaseCallback):
    """Custom callback to track wins/losses and print board info each episode."""
    
    def __init__(self, print_every: int = 100, verbose: int = 1):
        super().__init__(verbose)
        self.print_every = print_every
        self.episode_count = 0
        self.wins = 0
        self.losses = 0
        self.pushes = 0
        self.total_reward = 0.0
        
    def _on_step(self) -> bool:
        # Check if episode ended - SB3 stores this in infos with 'terminal_observation'
        infos = self.locals.get("infos", [])
        
        for i, info in enumerate(infos):
            # When episode ends, SB3 puts 'terminal_observation' in info
            if "terminal_observation" in info or info.get("TimeLimit.truncated", False):
                self.episode_count += 1
                
                # Get reward - for terminal step, use the reward from this step
                rewards = self.locals.get("rewards", [0])
                reward = rewards[i] if i < len(rewards) else 0
                self.total_reward += reward
                
                if reward > 0:
                    self.wins += 1
                elif reward < 0:
                    self.losses += 1
                else:
                    self.pushes += 1
                
                # Print stats periodically
                if self.episode_count % self.print_every == 0:
                    win_rate = (self.wins / self.episode_count) * 100 if self.episode_count > 0 else 0
                    avg_reward = self.total_reward / self.episode_count if self.episode_count > 0 else 0
                    
                    print(f"\n{'='*50}")
                    print(f"Episode {self.episode_count}")
                    print(f"Result: {'WIN' if reward > 0 else 'LOSS' if reward < 0 else 'PUSH'} ({reward:+.2f})")
                    print(f"Stats: W:{self.wins} L:{self.losses} P:{self.pushes} | Win Rate: {win_rate:.1f}% | Avg Reward: {avg_reward:.2f}")
                    print(f"{'='*50}")
        
        return True


# ── hyper-parameters ─────────────────────────────────────────────────
# DEFAULTS = dict(
#     total_timesteps  = 200_000,   # how many env steps to train
#     n_steps          = 1024,       # rollout length before each update
#     batch_size       = 256,        # mini-batch size for SGD
#     n_epochs         = 10,        # gradient passes per rollout
#     learning_rate    = 1e-5,
#     gamma            = 0.95,      # discount factor
#     gae_lambda       = 0.9,      # GAE smoothing
#     clip_range       = 0.2,       # PPO clipping
#     ent_coef         = 0.05,      # entropy bonus (encourages exploration)
#     verbose          = 1,         # SB3 logging level
# )

# New coefficients
DEFAULTS = dict(
    total_timesteps  = 1_000_000,
    n_steps          = 2048,
    batch_size       = 256,
    n_epochs         = 10,
    learning_rate    = 3e-4,
    gamma            = 0.95,
    gae_lambda       = 0.9,
    clip_range       = 0.2,
    ent_coef         = 0.01,
    verbose          = 1,
)


def make_env():
    """Factory so DummyVecEnv can create the env."""
    return PokerEnv(buy_in=75.0, sb=5.0, bb=10.0)


def train(total_timesteps: int = DEFAULTS["total_timesteps"]):
    # ── 1. wrap env ──────────────────────────────────────────────────
    # DummyVecEnv is required by Stable Baselines even for 1 env.
    vec_env = DummyVecEnv([make_env])

    # ── 2. create PPO agent ──────────────────────────────────────────
    model = PPO(
        "MlpPolicy",                    # built-in 2-layer MLP policy
        vec_env,
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

    # ── 3. optional: eval callback (logs mean reward every 1 000 steps)
    eval_env  = DummyVecEnv([make_env])
    eval_cb   = EvalCallback(
        eval_env,
        eval_freq      = 1_000,
        n_eval_episodes = 20,
        verbose        = 1,
    )
    
    # ── 3b. win/loss tracking callback
    winloss_cb = WinLossCallback(print_every=100, verbose=1)

    # ── 4. train ─────────────────────────────────────────────────────
    print("=" * 60)
    print("  Starting PPO training …")
    print(f"  Total timesteps : {total_timesteps:,}")
    print("=" * 60)

    model.learn(total_timesteps=total_timesteps, callback=[eval_cb, winloss_cb])

    # ── 5. save the trained model ────────────────────────────────────
    model.save("poker_ppo_improved_player_model")
    print("\n✓  Model saved to  poker_ppo_improved_player_model.zip")

    # ── 6. quick evaluation after training ───────────────────────────
    print("\n── Post-training evaluation (100 hands) ──")
    env   = PokerEnv()
    wins  = losses = pushes = 0
    total_reward = 0.0

    for _ in range(100):
        obs, info = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(int(action))
        total_reward += reward
        if   reward > 0:  wins   += 1
        elif reward < 0:  losses += 1
        else:             pushes += 1

    print(f"  Wins   : {wins}")
    print(f"  Losses : {losses}")
    print(f"  Pushes : {pushes}")
    print(f"  Avg reward per hand : {total_reward / 100:.2f}")

    # ── 7. plot learning curve ───────────────────────────────────────
    _plot_learning_curve(eval_cb)

    return model


def _plot_learning_curve(eval_cb: EvalCallback):
    """Plot mean eval reward vs timestep."""
    results_plotter_available = False
    try:
        from stable_baselines3.common.results_plotter import plot_results
        results_plotter_available = True
    except ImportError:
        pass

    # Use the data stored inside the EvalCallback
    if not eval_cb.evaluations_results:
        print("  (no eval data to plot)")
        return

    timesteps = eval_cb.evaluations_timesteps
    mean_rewards = [np.mean(r) for r in eval_cb.evaluations_results]

    plt.figure(figsize=(10, 5))
    plt.plot(timesteps, mean_rewards, color="royalblue", linewidth=2)
    plt.fill_between(
        timesteps,
        [np.mean(r) - np.std(r) for r in eval_cb.evaluations_results],
        [np.mean(r) + np.std(r) for r in eval_cb.evaluations_results],
        alpha=0.2, color="royalblue",
    )
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Reward ($ per hand)")
    plt.title("PPO Poker Agent – Learning Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("learning_curve.png", dpi=150)
    plt.show()
    print("  ✓  Learning curve saved to learning_curve.png")


# ── CLI ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a PPO poker agent.")
    parser.add_argument(
        "--steps", type=int, default=DEFAULTS["total_timesteps"],
        help="Total environment timesteps to train (default 200 000)."
    )
    args = parser.parse_args()
    train(total_timesteps=args.steps)
