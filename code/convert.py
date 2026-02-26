# convert_snapshot.py — run once
import numpy as np
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv
from self_play_poker_env import SP_PokerEnv, make_masked_env

def make_env():
    return make_masked_env(SP_PokerEnv(buy_in=75.0, sb=5.0, bb=10.0))


for i in range(100_000, 1_000_001, 50_000):
    old = PPO.load(f"snapshots/snapshot_{i}")  # your old model path
    vec_env = DummyVecEnv([make_env])

    new = MaskablePPO(
        "MlpPolicy", vec_env,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[256, 256, 256])),
    )

    # Copy weights directly — architecture is identical, only the policy class differs
    new.policy.load_state_dict(old.policy.state_dict())
    new.save(f"snapshots/snapshot_{i}")
    print(f"Done — use --resume snapshots/snapshot_{i}.zip")