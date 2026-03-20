import argparse
import os
import random

import numpy as np
import torch

from mccfr_environment import FixedPokerEnvironment
from MCCFR_train import StrategyNet, query_strategy_net

Env = FixedPokerEnvironment


def _deal_board(env: Env):
    if env._hand_over():
        return
    for _ in range(4):
        if not (hasattr(env.state, "can_deal_board") and env.state.can_deal_board()):
            break
        try:
            dealable = list(env.state.get_dealable_cards())
            n = env.state.board_dealing_count
            env.state.deal_board("".join(c.rank + c.suit for c in random.sample(dealable, n)))
        except Exception:
            break


def _legal_mask(env: Env, seat: int) -> list:
    return env._legal_mask(seat)


def _play_hand(env: Env, fn_seat0, fn_seat1) -> tuple:
    """Play one hand"""
    env.player_seat    = 0
    env._new_hand()
    env.last_actions   = [-1, -1]
    env.action_history = []

    fns = [fn_seat0, fn_seat1]

    for _ in range(100):
        _deal_board(env)
        if env._hand_over():
            break
        actor = env.state.actor_index
        if actor is None:
            break
        action = fns[actor](env, actor)
        ok = env._execute_action(action, player=actor)
        if ok:
            env.last_actions[actor] = action
            env.action_history.append((actor, action))
        if env._hand_over():
            break

    net0 = (float(env.state.stacks[0]) - float(env.buy_in)) / float(env.buy_in)
    return net0, -net0


def _cfr_fn(strategy_nets, device):
    def act(env: Env, seat: int) -> int:
        mask = _legal_mask(env, seat)
        return query_strategy_net(
            strategy_nets[seat], env.state, seat,
            env.action_history, mask, device, deterministic=False,
        )
    return act


def _random_fn(env: Env, seat: int) -> int:
    mask  = _legal_mask(env, seat)
    legal = [i for i, v in enumerate(mask) if v]
    return random.choice(legal) if legal else 1


def _ppo_fn(model):
    def act(env: Env, seat: int) -> int:
        obs  = env._build_obs(seat)
        arr  = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        mask = _legal_mask(env, seat)
        action, _ = model.predict(
            arr, action_masks=np.array(mask, dtype=bool), deterministic=False,
        )
        return int(action[0]) if hasattr(action, "__len__") else int(action)
    return act



class Stats:
    def __init__(self):
        self.wins = self.losses = self.pushes = 0
        self.chip_sum = 0.0

    def record(self, net: float):
        if   net > 0: self.wins   += 1
        elif net < 0: self.losses += 1
        else:         self.pushes += 1
        self.chip_sum += net

    def total(self) -> int:
        return self.wins + self.losses + self.pushes

    def win_pct(self) -> float:
        n = self.total()
        return self.wins / n * 100 if n else 0.0

    def avg_chips(self, buy_in: float = 100.0) -> float:
        n = self.total()
        return self.chip_sum / n * buy_in if n else 0.0


def evaluate(label: str, cfr_fn, opponent_fn, n_hands: int, buy_in: float = 100.0):
    """Run n_hands alternating seats and give a summary report with statistics"""
    env    = FixedPokerEnvironment(buy_in=buy_in, sb=5.0, bb=10.0)
    env._opponent_model = None

    cfr_stats  = Stats()
    opp_stats  = Stats()
    cfr_bb     = Stats()   
    cfr_sb     = Stats()   
    print(f"\n{'─'*60}")
    print(f"  {label}  ({n_hands:,} hands)")
    print(f"{'─'*60}")

    for idx in range(n_hands):
        if idx % 2 == 0:
           
            net_cfr, net_opp = _play_hand(env, cfr_fn, opponent_fn)
            cfr_bb.record(net_cfr)
        else:
            
            net_opp, net_cfr = _play_hand(env, opponent_fn, cfr_fn)
            cfr_sb.record(net_cfr)

        cfr_stats.record(net_cfr)
        opp_stats.record(net_opp)

        if (idx + 1) % max(1, n_hands // 10) == 0:
            pct = (idx + 1) / n_hands * 100
            print(f"  [{pct:5.1f}%]  CFR win%: {cfr_stats.win_pct():.1f}%  "
                  f"avg chips: {cfr_stats.avg_chips(buy_in):+.2f}")

    W = 60
    print(f"\n{'='*W}")
    print(f"  RESULT: CFR vs {label}")
    print(f"{'─'*W}")
    print(f"  {'':28}  {'Wins':>5}  {'Losses':>6}  {'Win%':>6}  {'Avg chips':>10}")
    print(f"  {'─'*(W-2)}")
    for lbl, st in [("CFR (overall)", cfr_stats),
                    ("CFR as BB (seat 0)", cfr_bb),
                    ("CFR as SB (seat 1)", cfr_sb),
                    ("Opponent (overall)", opp_stats)]:
        print(f"  {lbl:<28}  {st.wins:>5}  {st.losses:>6}  "
              f"{st.win_pct():>5.1f}%  {st.avg_chips(buy_in):>+10.2f}")
    print(f"{'='*W}")

    return cfr_stats



def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Deep CFR policy."
    )
    parser.add_argument("--cfr-model", required=True,
                        help="Path to cfr_model.pt (or a checkpoint .pt file).")
    parser.add_argument("--vs", required=True,
                        choices=["random", "ppo", "self", "all"],
                        help="Opponent type to evaluate against.")
    parser.add_argument("--ppo-model", default=None,
                        help="Path to PPO .zip (required when --vs ppo or --vs all).")
    parser.add_argument("--hands", type=int, default=2000,
                        help="Hands per evaluation (default: 2000).")
    parser.add_argument("--buy-in", type=float, default=100.0,
                        help="Starting stack (default: 100).")
    args = parser.parse_args()

    if not os.path.exists(args.cfr_model):
        raise SystemExit(f"CFR model not found: {args.cfr_model}")

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.cfr_model, map_location=device)
    strategy_nets = []
    for p in range(2):
        net = StrategyNet().to(device).eval()
        net.load_state_dict(checkpoint[f"strategy_net_{p}"])
        strategy_nets.append(net)
    cfr_fn = _cfr_fn(strategy_nets, device)
    print(f"CFR model loaded: {args.cfr_model}  (device: {device})")

    modes = ["random", "ppo", "self"] if args.vs == "all" else [args.vs]

    for mode in modes:
        if mode == "random":
            evaluate("Random opponent", cfr_fn, _random_fn, args.hands, args.buy_in)

        elif mode == "ppo":
            if not args.ppo_model:
                print("  [skip] --ppo-model required for --vs ppo")
                continue
            if not os.path.exists(args.ppo_model) and not os.path.exists(args.ppo_model + ".zip"):
                print(f"  [skip] PPO model not found: {args.ppo_model}")
                continue
            from sb3_contrib import MaskablePPO
            ppo = MaskablePPO.load(args.ppo_model)
            evaluate(f"PPO  ({os.path.basename(args.ppo_model)})",
                     cfr_fn, _ppo_fn(ppo), args.hands, args.buy_in)

        elif mode == "self":
            evaluate("CFR self-play (Nash check)", cfr_fn, cfr_fn, args.hands, args.buy_in)


if __name__ == "__main__":
    main()
