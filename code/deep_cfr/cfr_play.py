import argparse
import os
import random
from typing import List

import numpy as np
import torch

from mccfr_environment import FixedPokerEnvironment
from MCCFR_train import StrategyNet, query_strategy_net

Env = FixedPokerEnvironment

ACTION_NAMES = {
    0: "FOLD",
    1: "CHECK",
    2: "CALL",
    3: "RAISE (quarter pot)",
    4: "RAISE (half pot)",
    5: "RAISE (full pot)",
}



def hand_over(env: Env) -> bool:
    if env.state.actor_index is not None:
        return False
    if hasattr(env.state, "can_deal_board") and env.state.can_deal_board():
        return False
    return True


def betting_round(env: Env) -> int:
    n = len(env.state.board_cards) if env.state.board_cards else 0
    if n == 0:   return 0
    elif n == 3: return 1
    elif n == 4: return 2
    else:        return 3


def random_cards(env: Env, n: int) -> str:
    dealable = list(env.state.get_dealable_cards())
    cards = random.sample(dealable, n)
    return "".join(c.rank + c.suit for c in cards)


def deal_board_if_needed(env: Env):
    if hand_over(env):
        return
    done = 0
    while hasattr(env.state, "can_deal_board") and env.state.can_deal_board():
        if done >= 4:
            break
        try:
            n = env.state.board_dealing_count
            env.state.deal_board(random_cards(env, n))
            done += 1
        except Exception:
            break


def resolve_hand(env: Env, human_seat: int) -> float:
    return (float(env.state.stacks[human_seat]) - float(env.buy_in)) / float(env.buy_in)


def flatten_board(state) -> List[str]:
    cards = []
    if not state.board_cards:
        return cards
    for part in state.board_cards:
        if not part:
            continue
        if isinstance(part, (list, tuple)):
            cards.extend(c.rank + c.suit for c in part)
        else:
            cards.append(part.rank + part.suit)
    return cards


def get_total_pot(env: Env) -> float:
    try:
        return float(env.state.total_pot_amount)
    except Exception:
        return 0.0


def get_call_amount(env: Env, player: int) -> float:
    try:
        bets    = list(env.state.bets)
        my_bet  = float(bets[player])     if player       < len(bets) else 0.0
        opp_bet = float(bets[1 - player]) if 1 - player   < len(bets) else 0.0
        return max(0.0, opp_bet - my_bet)
    except Exception:
        return 0.0


def get_raise_amounts(env: Env, player: int) -> dict:
    pot   = get_total_pot(env)
    stack = float(env.state.stacks[player])
    return {
        Env.RAISE_QUARTER: min(pot * 0.25, stack),
        Env.RAISE_HALF:    min(pot * 0.50, stack),
        Env.RAISE_POT:     min(pot * 1.00, stack),
    }


def compute_legal_mask(env: Env, player: int) -> list:
    mask = [0, 0, 0, 0, 0, 0]
    if hand_over(env):
        return mask
    mask[Env.FOLD] = 1
    try:
        if env.state.can_check_or_call():
            bets    = list(env.state.bets)
            my_bet  = float(bets[player])     if player       < len(bets) else 0.0
            opp_bet = float(bets[1 - player]) if 1 - player   < len(bets) else 0.0
            if opp_bet > my_bet:
                mask[Env.CALL]  = 1
            else:
                mask[Env.CHECK] = 1
                mask[Env.FOLD]  = 0
        if env.state.can_complete_bet_or_raise_to():
            mask[Env.RAISE_QUARTER] = 1
            mask[Env.RAISE_HALF]    = 1
            mask[Env.RAISE_POT]     = 1
    except Exception as e:
        print(f"  [legal_mask warning] {e}")
        mask[Env.CHECK] = 1
        mask[Env.FOLD]  = 0
    return mask


def execute_action(env: Env, action: int, player: int) -> bool:
    """Execute action on in pokerkit"""
    if action == Env.FOLD:
        env.state.fold()
    elif action in (Env.CHECK, Env.CALL):
        env.state.check_or_call()
    elif action in (Env.RAISE_QUARTER, Env.RAISE_HALF, Env.RAISE_POT):
        pot         = get_total_pot(env)
        stack       = float(env.state.stacks[player])
        mult        = {Env.RAISE_QUARTER: 0.25, Env.RAISE_HALF: 0.50, Env.RAISE_POT: 1.00}[action]
        current_bet = float(env.state.bets[player]) if player < len(env.state.bets) else 0.0
        target      = min(current_bet + pot * mult, stack + current_bet)
        min_raise   = env.state.min_completion_betting_or_raising_to_amount
        if min_raise is not None:
            target = max(target, float(min_raise))
        env.state.complete_bet_or_raise_to(target)

    env.last_actions[player] = action
    env.action_history.append((player, action))  
    deal_board_if_needed(env)
    return hand_over(env)



def show_table(env: Env, human_seat: int):
    print("\n" + "=" * 55)
    hole = ["??", "??"]
    if env.state.hole_cards and env.state.hole_cards[human_seat]:
        hole = [c.rank + c.suit for c in env.state.hole_cards[human_seat]]
    print(f"  Your hole cards : {hole[0]}  {hole[1]}")

    board = flatten_board(env.state)
    print(f"  Board           : {' '.join(board) if board else '(none)'}")

    try:
        round_bets = list(env.state.bets)
    except Exception:
        round_bets = [0, 0]

    total_pot  = get_total_pot(env)
    your_stack = float(env.state.stacks[human_seat])
    opp_stack  = float(env.state.stacks[1 - human_seat])
    your_bet   = float(round_bets[human_seat])     if human_seat     < len(round_bets) else 0.0
    opp_bet    = float(round_bets[1 - human_seat]) if 1-human_seat   < len(round_bets) else 0.0
    to_call    = get_call_amount(env, human_seat)

    print(f"  Pot (total)     : {total_pot:.1f}")
    print(f"  Your stack      : {your_stack:.1f}  (bet this round: {your_bet:.1f})")
    print(f"  Opp  stack      : {opp_stack:.1f}  (bet this round: {opp_bet:.1f})")
    if to_call > 0:
        print(f"  >>> To call     : {to_call:.1f} chips <<<")

    rounds   = ["Pre-flop", "Flop", "Turn", "River"]
    position = "BB (Big Blind)" if human_seat == 0 else "SB (Small Blind)"
    print(f"  Round           : {rounds[betting_round(env)]}")
    print(f"  Your position   : {position}")
    print("=" * 55)


def human_take_action(env: Env, human_seat: int) -> int:
    mask    = compute_legal_mask(env, human_seat)
    legal   = [i for i, v in enumerate(mask) if v]
    raises  = get_raise_amounts(env, human_seat)
    to_call = get_call_amount(env, human_seat)

    print("\n  Your options:")
    for a in legal:
        label = ACTION_NAMES.get(a, str(a))
        if a == Env.CALL:
            label += f"  [{to_call:.1f} chips]"
        elif a in raises:
            label += f"  [{raises[a]:.1f} chips]"
        print(f"    {a} → {label}")

    while True:
        choice = input("\n  Enter action number: ").strip()
        if choice.isdigit() and int(choice) in legal:
            return int(choice)
        print(f"  Invalid. Choose from: {legal}")


def cfr_take_action(strategy_nets, device, env: Env, actor: int) -> int:
    """get an action from Deep CFR"""
    mask   = compute_legal_mask(env, actor)
    return query_strategy_net(
        strategy_nets[actor],
        env.state,
        actor,
        env.action_history,
        mask,
        device,
        deterministic=False,
    )


def ppo_take_action(model, env: Env, actor: int) -> int:
    
    obs  = env._build_obs(actor)
    arr  = np.asarray(obs, dtype=np.float32).reshape(1, -1)
    mask = compute_legal_mask(env, actor)
    action, _ = model.predict(arr, action_masks=np.array(mask, dtype=bool), deterministic=True)
    return int(action[0]) if isinstance(action, (list, tuple, np.ndarray)) else int(action)


def show_result(env: Env, human_seat: int, opp_cards: list):
    board = flatten_board(env.state)
    print("\n" + "─" * 55)
    if board:
        print(f"  Final board : {' '.join(board)}")
    try:
        your_hole = [c.rank + c.suit for c in env.state.hole_cards[human_seat]]
        if your_hole: print(f"  Your hand   : {' '.join(your_hole)}")
    except Exception:
        pass
    if opp_cards:
        print(f"  Opp  hand   : {' '.join(opp_cards)}  ← revealed")

    human_net = resolve_hand(env, human_seat)
    print("\n" + "=" * 55)
    if human_net > 0:
        print(f"  YOU WIN    +{human_net * env.buy_in:.1f} chips")
    elif human_net < 0:
        print(f"  YOU LOSE    {human_net * env.buy_in:.1f} chips")
    else:
        print("  PUSH (tie)")
    print(f"  Stacks → You: {env.state.stacks[human_seat]:.1f} | Opp: {env.state.stacks[1-human_seat]:.1f}")
    print("=" * 55)



def play_hand(ai_fn, human_seat: int):
    """Play one hand"""
    env = FixedPokerEnvironment(buy_in=100.0, sb=5.0, bb=10.0)
    env.player_seat  = human_seat
    env._new_hand()
    env.last_actions   = [-1, -1]
    env.action_history = []

    opp_str  = env.opponents_cards         
    opp_cards = [opp_str[i:i+2] for i in range(0, len(opp_str), 2)] if opp_str else []

    print(f"\n{'─' * 55}")
    print(f"  NEW HAND  |  Buy-in: {env.buy_in:.0f}  SB: {env.sb:.0f}  BB: {env.bb:.0f}")
    print(f"  You are seat {human_seat} ({'BB' if human_seat == 0 else 'SB'})")
    print(f"  Blinds posted → Pot: {get_total_pot(env):.1f}")

    while True:
        deal_board_if_needed(env)

        if hand_over(env):
            show_result(env, human_seat, opp_cards)
            return

        actor = env.state.actor_index
        if actor is None:
            deal_board_if_needed(env)
            if hand_over(env):
                show_result(env, human_seat, opp_cards)
            return

        if actor == human_seat:
            show_table(env, human_seat)
            action = human_take_action(env, human_seat)
            print(f"\n  You play : {ACTION_NAMES.get(action, action)}")
            done = execute_action(env, action, human_seat)
            if done:
                show_result(env, human_seat, opp_cards)
                return

        else:
            action     = ai_fn(env, actor)
            seat_label = "BB" if actor == 0 else "SB"

            if action in (Env.RAISE_QUARTER, Env.RAISE_HALF, Env.RAISE_POT):
                chip_amt = get_raise_amounts(env, actor).get(action, 0.0)
                print(f"\n  AI ({seat_label}) : {ACTION_NAMES[action]}  [{chip_amt:.1f} chips]")
            elif action == Env.CALL:
                print(f"\n  AI ({seat_label}) : CALL  [{get_call_amount(env, actor):.1f} chips]")
            else:
                print(f"\n  AI ({seat_label}) : {ACTION_NAMES.get(action, action)}")

            done = execute_action(env, action, actor)
            if action in (Env.RAISE_QUARTER, Env.RAISE_HALF, Env.RAISE_POT):
                print(f"  → Pot now {get_total_pot(env):.1f}  |  Your call: {get_call_amount(env, human_seat):.1f}")
            if done:
                show_result(env, human_seat, opp_cards)
                return


def main():
    parser = argparse.ArgumentParser(description="Play poker against a trained AI.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--cfr-model", metavar="PATH",
        help="Path to a Deep CFR .pt checkpoint (cfr_model.pt or cfr_model_ckptN.pt).",
    )
    group.add_argument(
        "--ppo-model", metavar="PATH",
        help="Path to a MaskablePPO .zip snapshot.",
    )
    parser.add_argument(
        "--human-seat", type=int, default=0, choices=[0, 1],
        help="Your seat: 0=BB, 1=SB (default 0).",
    )
    parser.add_argument(
        "--hands", type=int, default=0,
        help="Number of hands to play (0 = unlimited).",
    )
    args = parser.parse_args()

    if args.cfr_model:
        if not os.path.exists(args.cfr_model):
            raise SystemExit(f"CFR model not found: {args.cfr_model}")

        device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(args.cfr_model, map_location=device)

        strategy_nets = []
        for p in range(2):
            net = StrategyNet().to(device).eval()
            net.load_state_dict(checkpoint[f"strategy_net_{p}"])
            strategy_nets.append(net)

        ai_fn = lambda env, actor: cfr_take_action(strategy_nets, device, env, actor)
        print(f"Deep CFR model loaded from {args.cfr_model}  (device: {device})")

    else:
        if not os.path.exists(args.ppo_model):
            raise SystemExit(f"PPO model not found: {args.ppo_model}")
        from sb3_contrib import MaskablePPO
        model = MaskablePPO.load(args.ppo_model)
        ai_fn = lambda env, actor: ppo_take_action(model, env, actor)
        print(f"PPO model loaded from {args.ppo_model}")

    print("Let's play!\n")

    played = 0
    while True:
        play_hand(ai_fn, args.human_seat)
        played += 1
        if args.hands > 0 and played >= args.hands:
            print(f"\nPlayed {played} hand(s). Goodbye!")
            break
        again = input("\nPlay another hand? [y/N]: ").strip().lower()
        if again != "y":
            print("Thanks for playing!")
            break


if __name__ == "__main__":
    main()
