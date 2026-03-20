import argparse
import os

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from maskable_environment import FixedPokerEnvironment


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))



def mask_fn(env):
    """gets legal mask"""
    inner = env
    while hasattr(inner, "env"):
        inner = inner.env
    return np.array(inner._legal_mask(inner.player_seat), dtype=bool)


def make_env(snapshot_dir_a: str, snapshot_dir_b: str, current_agent: str):
    """Creates an environemnt"""
    def _init():
        env = FixedPokerEnvironment(
            buy_in=100.0,
            sb=5.0,
            bb=10.0,
            snapshot_dir_a=snapshot_dir_a,
            snapshot_dir_b=snapshot_dir_b,
            current_agent=current_agent,
            opponent_refresh_freq=1000,
            snapshot_pool_size=20,
        )
        env = Monitor(env)
        return ActionMasker(env, mask_fn)
    return _init



class SnapshotCallback(BaseCallback):
    """Saves a model snapshot every set amount of steps"""

    def __init__(self, save_freq: int, snapshot_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.snapshot_dir = snapshot_dir
        os.makedirs(snapshot_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            path = os.path.join(self.snapshot_dir, f"snapshot_{self.num_timesteps}")
            self.model.save(path)
            if self.verbose:
                print(f"  [snapshot] saved → {path}.zip")
        return True


class WinLossCallback(BaseCallback):
    """Prints rolling win/loss stats"""

    WINDOW = 500

    def __init__(self, agent_label: str = "?", print_every: int = 200, verbose: int = 1):
        super().__init__(verbose)
        self.agent_label = agent_label
        self.print_every = print_every
        self.episode_count = 0
        self.total_wins = 0
        self.total_losses = 0
        self.total_pushes = 0
        self._window: list[float] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep_info = info.get("episode")
            if ep_info is None:
                continue

            reward = float(ep_info["r"])
            self.episode_count += 1

            if reward > 0:
                self.total_wins += 1
            elif reward < 0:
                self.total_losses += 1
            else:
                self.total_pushes += 1

            self._window.append(reward)
            if len(self._window) > self.WINDOW:
                self._window.pop(0)

            if self.episode_count % self.print_every == 0:
                n = len(self._window)
                rw = sum(1 for r in self._window if r > 0) / n * 100
                avg = sum(self._window) / n
                print(
                    f"\n  {'─'*55}\n"
                    f"  [Agent {self.agent_label}] ep {self.episode_count:>6} | "
                    f"W {self.total_wins} L {self.total_losses} P {self.total_pushes} | "
                    f"rolling win% {rw:.1f}% | rolling avg {avg:+.3f}  "
                    f"(last {n} hands)\n"
                    f"  {'─'*55}"
                )
        return True



DEFAULTS = dict(
    n_steps=4096,
    batch_size=512,
    n_epochs=6,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    verbose=1,
)


def _build_model(vec_env, agent_label: str) -> MaskablePPO:
    return MaskablePPO(
        "MlpPolicy",
        vec_env,
        device="cuda",
        policy_kwargs=dict(net_arch=[256, 256]),
        n_steps=DEFAULTS["n_steps"],
        batch_size=DEFAULTS["batch_size"],
        n_epochs=DEFAULTS["n_epochs"],
        learning_rate=DEFAULTS["learning_rate"],
        gamma=DEFAULTS["gamma"],
        gae_lambda=DEFAULTS["gae_lambda"],
        clip_range=DEFAULTS["clip_range"],
        ent_coef=DEFAULTS["ent_coef"],
        vf_coef=DEFAULTS["vf_coef"],
        verbose=DEFAULTS["verbose"],
        tensorboard_log=os.path.join(SCRIPT_DIR, f"tensorboard_{agent_label}"),
    )



def train_fsp(
    rounds: int,
    steps_per_round: int,
    snapshot_freq: int,
    snapshot_dir: str,
    resume_a: str | None = None,
    resume_b: str | None = None,
):
    """training loop"""
    snap_dir_a = os.path.join(snapshot_dir, "A")
    snap_dir_b = os.path.join(snapshot_dir, "B")
    os.makedirs(snap_dir_a, exist_ok=True)
    os.makedirs(snap_dir_b, exist_ok=True)

    print("=" * 60)
    print("  MaskablePPO — Fictitious Self-Play (FSP)")
    print(f"  OBS dim        : {FixedPokerEnvironment.OBS_DIM}")
    print(f"  Action history : {FixedPokerEnvironment.ACTION_HISTORY_LEN} slots")
    print(f"  FSP rounds     : {rounds}")
    print(f"  Steps/round    : {steps_per_round:,}")
    print(f"  Snapshot freq  : {snapshot_freq:,}")
    print(f"  Snapshot dir A : {snap_dir_a}")
    print(f"  Snapshot dir B : {snap_dir_b}")
    print("=" * 60)

   
    def _make_vec_env(agent_label: str):
        return DummyVecEnv([make_env(snap_dir_a, snap_dir_b, current_agent=agent_label)])

    models: dict[str, MaskablePPO] = {}
    snap_dirs = {"A": snap_dir_a, "B": snap_dir_b}
    resumes = {"A": resume_a, "B": resume_b}

    for label in ("A", "B"):
        vec_env = _make_vec_env(label)
        resume = resumes[label]
        if resume and (os.path.exists(resume) or os.path.exists(resume + ".zip")):
            print(f"  Resuming agent {label} from: {resume}")
            models[label] = MaskablePPO.load(
                resume,
                env=vec_env,
                tensorboard_log=os.path.join(SCRIPT_DIR, f"tensorboard_{label}"),
            )
            models[label].ent_coef = DEFAULTS["ent_coef"]
        else:
            models[label] = _build_model(vec_env, label)

    for round_idx in range(rounds):
        for label in ("A", "B"):
            print(f"\n[FSP round {round_idx + 1}/{rounds}] Training agent {label} ...")

            vec_env = _make_vec_env(label)
            models[label].set_env(vec_env)

            snap_cb = SnapshotCallback(
                save_freq=snapshot_freq,
                snapshot_dir=snap_dirs[label],
                verbose=1,
            )
            winloss_cb = WinLossCallback(
                agent_label=label,
                print_every=200,
                verbose=1,
            )

            models[label].learn(
                total_timesteps=steps_per_round,
                callback=[snap_cb, winloss_cb],
                reset_num_timesteps=False,  
            )

          
            round_ts = (round_idx + 1) * steps_per_round
            end_snap = os.path.join(snap_dirs[label], f"snapshot_{round_ts}")
            models[label].save(end_snap)
            print(f"  [FSP] Agent {label} end-of-round snapshot → {end_snap}.zip")

   
    for label in ("A", "B"):
        final_path = os.path.join(SCRIPT_DIR, f"poker_fsp_final_{label}")
        models[label].save(final_path)
        print(f"\n  Final model {label} saved → {final_path}.zip")

    return models



def evaluate(model_a: MaskablePPO, model_b: MaskablePPO, n_hands: int = 100):
    """Play agent A vs agent B and collect statistics"""
    print(f"\n── A vs B evaluation ({n_hands} hands) ──")

    snap_dir_a = "./eval_snaps/A"
    snap_dir_b = "./eval_snaps/B"

    env = FixedPokerEnvironment(
        buy_in=100.0, sb=5.0, bb=10.0,
        snapshot_dir_a=snap_dir_a,
        snapshot_dir_b=snap_dir_b,
    )
    env._opponent_model = model_b  

    masked_env = ActionMasker(env, mask_fn)

    a_wins = a_losses = pushes = 0
    total_reward = 0.0

    for _ in range(n_hands):
        obs, info = masked_env.reset()
        done = False
        step_limit = 0
        reward = 0.0
        while not done and step_limit < 100:
            masks = np.array(info.get("action_masks", mask_fn(masked_env)), dtype=bool)
            action, _ = model_a.predict(obs, action_masks=masks, deterministic=True)
            obs, reward, done, truncated, info = masked_env.step(int(action))
            step_limit += 1
        total_reward += reward
        if reward > 0:
            a_wins += 1
        elif reward < 0:
            a_losses += 1
        else:
            pushes += 1

    print(f"  A wins {a_wins} | A losses {a_losses} | Pushes {pushes}")
    print(f"  A win rate : {a_wins / n_hands * 100:.1f}%")
    print(f"  A avg reward/hand : {total_reward / n_hands:+.3f}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train two MaskablePPO poker agents via Fictitious Self-Play."
    )
    parser.add_argument(
        "--rounds", type=int, default=60,
        help="Number of FSP rounds (default 60). Each round trains both A and B."
    )
    parser.add_argument(
        "--steps-per-round", type=int, default=100_000,
        help="Timesteps each agent trains per round (default 100 000)."
    )
    parser.add_argument(
        "--snapshot-freq", type=int, default=50_000,
        help="Save a snapshot every N timesteps within a round (default 50 000)."
    )
    parser.add_argument(
        "--snapshot-dir", type=str, default=os.path.join(SCRIPT_DIR, "fsp_snaps"),
        help="Root directory for snapshots; A/ and B/ subdirs created automatically."
    )
    parser.add_argument(
        "--resume-a", type=str, default=None,
        help="Path to a .zip model to resume agent A from."
    )
    parser.add_argument(
        "--resume-b", type=str, default=None,
        help="Path to a .zip model to resume agent B from."
    )
    parser.add_argument(
        "--eval-hands", type=int, default=100,
        help="Number of evaluation hands (A vs B) after training (default 100)."
    )
    args = parser.parse_args()

    models = train_fsp(
        rounds=args.rounds,
        steps_per_round=args.steps_per_round,
        snapshot_freq=args.snapshot_freq,
        snapshot_dir=args.snapshot_dir,
        resume_a=args.resume_a,
        resume_b=args.resume_b,
    )

    if args.eval_hands > 0:
        evaluate(models["A"], models["B"], n_hands=args.eval_hands)
