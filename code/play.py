"""
Interactive human-vs-PPO model play script.

Usage:
    python play.py --model snapshots/final_1000000.zip --human-seat 0
"""

import argparse
import os
from typing import List

import numpy as np
from stable_baselines3 import PPO

from self_play_poker_env import SP_PokerEnv


ACTION_NAMES = {
    0: "FOLD",
    1: "CHECK",
    2: "CALL",
    3: "RAISE (quarter pot)",
    4: "RAISE (half pot)",
    5: "RAISE (full pot)",
}


def flatten_board(state) -> List[str]:
    cards = []
    if not state.board_cards:
        return cards
    for part in state.board_cards:
        if not part:
            continue
        if isinstance(part, (list, tuple)):
            for c in part:
                cards.append(c.rank + c.suit)
        else:
            cards.append(part.rank + part.suit)
    return cards


def get_total_pot(env: SP_PokerEnv) -> float:
    try:
        return float(env.state.total_pot_amount)
    except Exception:
        return 0.0


def get_call_amount(env: SP_PokerEnv, player: int) -> float:
    try:
        bets    = list(env.state.bets)
        my_bet  = float(bets[player])   if player   < len(bets) else 0.0
        opp_bet = float(bets[1-player]) if 1-player < len(bets) else 0.0
        return max(0.0, opp_bet - my_bet)
    except Exception:
        return 0.0


def get_raise_amounts(env: SP_PokerEnv, player: int) -> dict:
    pot   = get_total_pot(env)
    stack = float(env.state.stacks[player])
    return {
        SP_PokerEnv.RAISE_QUARTER: min(pot * 0.25, stack),
        SP_PokerEnv.RAISE_HALF:    min(pot * 0.50, stack),
        SP_PokerEnv.RAISE_POT:     min(pot * 1.00, stack),
    }


def compute_legal_mask(env: SP_PokerEnv, player: int) -> list:
    mask = [0, 0, 0, 0, 0, 0]
    if env._hand_over():
        return mask  # all zeros — hand is done, no actions valid

    mask[SP_PokerEnv.FOLD] = 1
    try:
        if env.state.can_check_or_call():
            to_call = get_call_amount(env, player)
            if to_call > 0:
                mask[SP_PokerEnv.CALL] = 1
            else:
                mask[SP_PokerEnv.CHECK] = 1
        if env.state.can_complete_bet_or_raise_to():
            mask[SP_PokerEnv.RAISE_QUARTER] = 1
            mask[SP_PokerEnv.RAISE_HALF]    = 1
            mask[SP_PokerEnv.RAISE_POT]     = 1
    except Exception as e:
        print(f"  [legal_mask warning] {e}")
        mask[SP_PokerEnv.CHECK] = 1

    return mask


def execute_action_safely(env: SP_PokerEnv, action: int, player: int) -> bool:
    """
    Execute an action directly against PokerKit state.
    Returns True if the hand is now over (fold/terminal), False otherwise.
    This bypasses env._execute_action entirely to avoid any silent swallowing.
    """
    
    if action == SP_PokerEnv.FOLD:
        env.state.fold()
        env.last_actions[player] = action
        return True  # folding always ends the hand

    elif action in (SP_PokerEnv.CHECK, SP_PokerEnv.CALL):
        env.state.check_or_call()
        env.last_actions[player] = action

    elif action in (SP_PokerEnv.RAISE_QUARTER, SP_PokerEnv.RAISE_HALF, SP_PokerEnv.RAISE_POT):
        pot   = get_total_pot(env)
        stack = float(env.state.stacks[player])
        mult  = {
            SP_PokerEnv.RAISE_QUARTER: 0.25,
            SP_PokerEnv.RAISE_HALF:    0.50,
            SP_PokerEnv.RAISE_POT:     1.00,
        }[action]
        current_bet = float(env.state.bets[player]) if player < len(env.state.bets) else 0.0
        target      = current_bet + pot * mult
        target      = min(target, stack + current_bet)
        min_raise   = env.state.min_completion_betting_or_raising_to_amount
        if min_raise is not None:
            target = max(target, float(min_raise))
        env.state.complete_bet_or_raise_to(target)
        env.last_actions[player] = action

    

    # Deal board if a street just ended
    env._deal_board_if_needed()
    return env._hand_over()


def show_table(env: SP_PokerEnv, human_seat: int):
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
    opp_stack  = float(env.state.stacks[1-human_seat])
    your_bet   = float(round_bets[human_seat])   if human_seat   < len(round_bets) else 0.0
    opp_bet    = float(round_bets[1-human_seat]) if 1-human_seat < len(round_bets) else 0.0
    to_call    = get_call_amount(env, human_seat)

    print(f"  Pot (total)     : {total_pot:.1f}")
    print(f"  Your stack      : {your_stack:.1f}  (bet this round: {your_bet:.1f})")
    print(f"  Opp  stack      : {opp_stack:.1f}  (bet this round: {opp_bet:.1f})")
    if to_call > 0:
        print(f"  >>> To call     : {to_call:.1f} chips <<<")

    rounds   = ["Pre-flop", "Flop", "Turn", "River"]
    position = "SB (Small Blind)" if human_seat == 0 else "BB (Big Blind)"
    print(f"  Round           : {rounds[env._betting_round()]}")
    print(f"  Your position   : {position}")
    print("=" * 55)


def human_take_action(env: SP_PokerEnv, human_seat: int) -> int:
    mask    = compute_legal_mask(env, human_seat)
    legal   = [i for i, v in enumerate(mask) if v]
    raises  = get_raise_amounts(env, human_seat)
    to_call = get_call_amount(env, human_seat)

    print("\n  Your options:")
    for a in legal:
        label = ACTION_NAMES.get(a, str(a))
        if a == SP_PokerEnv.CALL:
            label += f"  [{to_call:.1f} chips]"
        elif a in raises:
            label += f"  [{raises[a]:.1f} chips]"
        print(f"    {a} → {label}")

    while True:
        choice = input("\n  Enter action number: ").strip()
        if choice.isdigit() and int(choice) in legal:
            return int(choice)
        print(f"  Invalid. Choose from: {legal}")


def model_take_action(model: PPO, env: SP_PokerEnv, actor: int) -> int:
    obs = env._build_obs(player=actor)
    arr = np.asarray(obs, dtype=np.float32).reshape(1, -1)
    action, _ = model.predict(arr, deterministic=True)
    action = int(action[0]) if isinstance(action, (list, tuple, np.ndarray)) else int(action)

    mask = compute_legal_mask(env, actor)
    if not mask[action]:
        legal = [i for i, v in enumerate(mask) if v]
        action = legal[0] if legal else SP_PokerEnv.CHECK
    return action


def show_result(env: SP_PokerEnv, human_seat: int):
    board = flatten_board(env.state)
    print("\n" + "─" * 55)
    if board:
        print(f"  Final board : {' '.join(board)}")

    # try:
    your_hole = [c.rank + c.suit for c in env.state.hole_cards[human_seat]]   #if env.state.hole_cards and env.state.hole_cards[human_seat]   else []
    opp_hole  = [c.rank + c.suit for c in env.state.hole_cards[1-human_seat]] #if env.state.hole_cards and env.state.hole_cards[1-human_seat] else []
    if your_hole: print(f"  Your hand   : {' '.join(your_hole)}")
    if opp_hole:  print(f"  Opp  hand   : {' '.join(opp_hole)}")
    # except Exception:
    #     pass

    res       = env._resolve_hand()
    human_net = res if human_seat == 0 else -res

    print("\n" + "=" * 55)
    if human_net > 0:
        print(f"  ✓ YOU WIN    +{human_net * env.buy_in:.1f} chips")
    elif human_net < 0:
        print(f"  ✗ YOU LOSE    {human_net * env.buy_in:.1f} chips")
    else:
        print("  — PUSH (tie)")
    print(f"  Stacks → You: {env.state.stacks[human_seat]:.1f} | Opp: {env.state.stacks[1-human_seat]:.1f}")
    print("=" * 55)


def play_hand(model: PPO, human_seat: int):
    env = SP_PokerEnv(buy_in=75.0, sb=5.0, bb=10.0)
    env._new_hand()
    env.last_actions = [-1, -1]

    print(f"\n{'─' * 55}")
    print(f"  NEW HAND  |  Buy-in: {env.buy_in:.0f}  SB: {env.sb:.0f}  BB: {env.bb:.0f}")
    print(f"  You are seat {human_seat} ({'SB' if human_seat == 0 else 'BB'})")
    print(f"  Blinds posted → Pot: {get_total_pot(env):.1f}")

    while True:
        # Always deal any pending board cards first
        env._deal_board_if_needed()

        # Check if hand ended
        if env._hand_over():
            show_result(env, human_seat)
            return

        actor = env.state.actor_index

        # actor is None but hand not over = need more board cards
        if actor is None:
            env._deal_board_if_needed()
            if env._hand_over():
                show_result(env, human_seat)
            return

        if actor == human_seat:
            # ── Human acts ────────────────────────────────────
            show_table(env, human_seat)
            action  = human_take_action(env, human_seat)
            print(f"\n  You play : {ACTION_NAMES.get(action, action)}")
            hand_over = execute_action_safely(env, action, human_seat)
            if hand_over:
                show_result(env, human_seat)
                return

        else:
            # ── Model acts ─────────────────────────────────────
            action     = model_take_action(model, env, actor)
            seat_label = "SB" if actor == 0 else "BB"

            if action in (SP_PokerEnv.RAISE_QUARTER, SP_PokerEnv.RAISE_HALF, SP_PokerEnv.RAISE_POT):
                chip_amt = get_raise_amounts(env, actor).get(action, 0)
                print(f"\n  Model ({seat_label}) : {ACTION_NAMES[action]}  [{chip_amt:.1f} chips]")
            elif action == SP_PokerEnv.CALL:
                print(f"\n  Model ({seat_label}) : CALL  [{get_call_amount(env, actor):.1f} chips]")
            else:
                print(f"\n  Model ({seat_label}) : {ACTION_NAMES.get(action, action)}")

            hand_over = execute_action_safely(env, action, actor)

            # Show updated pot after model raises so human sees it before their turn
            if action in (SP_PokerEnv.RAISE_QUARTER, SP_PokerEnv.RAISE_HALF, SP_PokerEnv.RAISE_POT):
                print(f"  → Pot now {get_total_pot(env):.1f}  |  Your call: {get_call_amount(env, human_seat):.1f}")

            if hand_over:
                show_result(env, human_seat)
                return


def main():
    parser = argparse.ArgumentParser(description="Play poker against a trained PPO model.")
    parser.add_argument("--model",      required=True,  help="Path to PPO model .zip")
    parser.add_argument("--human-seat", type=int, default=0, choices=[0, 1],
                        help="Your seat: 0=SB, 1=BB (default 0)")
    parser.add_argument("--hands",      type=int, default=0,
                        help="Number of hands to play (0 = unlimited)")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f"Model not found: {args.model}")

    print(f"Loading model from {args.model} ...")
    model = PPO.load(args.model)
    print("Model loaded. Let's play!\n")

    played = 0
    while True:
        play_hand(model, args.human_seat)
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