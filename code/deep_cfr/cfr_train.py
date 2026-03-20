import argparse
import copy
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from pokerkit import Automation, NoLimitTexasHoldem

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

FOLD          = 0
CHECK         = 1
CALL          = 2
RAISE_QUARTER = 3
RAISE_HALF    = 4
RAISE_POT     = 5
NUM_ACTIONS   = 6

BUY_IN = 100.0
SB     = 5.0
BB     = 10.0
DECK   = [r + s for r in "23456789TJQKA" for s in "cdhs"]

RANKS             = "23456789TJQKA"
SUITS             = "cdhs"
RANK_TO_IDX       = {r: i + 1 for i, r in enumerate(RANKS)}
SUIT_TO_IDX       = {s: j     for j, s in enumerate(SUITS)}
OBS_DIM           = 85        
ACTION_HISTORY_LEN = 6


def _encode_card(card: str) -> np.ndarray:
    vec      = np.zeros(5, dtype=np.float32)
    vec[0]   = RANK_TO_IDX[card[0]] / 13.0
    vec[1 + SUIT_TO_IDX[card[1]]] = 1.0
    return vec


def _cards_to_vec(cards: list) -> np.ndarray:
    if not cards:
        return np.zeros(5, dtype=np.float32)
    return np.concatenate([_encode_card(c) for c in cards])


def build_obs(state, player: int, action_history: list) -> np.ndarray:
    """Build the 85 dimensional  observation vector"""
    obs            = np.zeros(OBS_DIM, dtype=np.float32)
    hole_cards     = []
    community_cards = []

    if state.hole_cards and state.hole_cards[player]:
        hole_cards = [c.rank + c.suit for c in state.hole_cards[player]]

    if state.board_cards:
        for part in state.board_cards:
            if not part:
                continue
            if isinstance(part, (list, tuple)):
                community_cards.extend(c.rank + c.suit for c in part)
            else:
                community_cards.append(part.rank + part.suit)

    obs[0:10]  = _cards_to_vec(hole_cards)
    obs[10:35] = 0.0
    board_vec  = _cards_to_vec(community_cards)
    obs[10: 10 + len(board_vec)] = board_vec

    obs[35] = 0.0 if player == 0 else 1.0
    obs[36] = float(state.stacks[player])     / BUY_IN
    obs[37] = float(state.stacks[1 - player]) / BUY_IN
    obs[38] = float(state.total_pot_amount)   / (2 * BUY_IN)

    n_board  = len(state.board_cards) if state.board_cards else 0
    si       = 0 if n_board == 0 else (1 if n_board == 3 else (2 if n_board == 4 else 3))
    obs[39 + si] = 1.0

    history = action_history[-ACTION_HISTORY_LEN:]
    for i, (actor, act) in enumerate(history):
        slot       = 43 + i * 7
        obs[slot]  = 0.0 if actor == player else 1.0
        if 0 <= act < 6:
            obs[slot + 1 + act] = 1.0

    return obs



def create_state(hole0: str, hole1: str):
    """Create a pokerkit state"""
    state = NoLimitTexasHoldem.create_state(
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
        True, 0, (SB, BB), BB, (BUY_IN, BUY_IN), 2,
    )
    state.deal_hole(hole0)
    state.deal_hole(hole1)
    return state


def is_terminal(state) -> bool:
    if state.actor_index is not None:
        return False
    if hasattr(state, "can_deal_board") and state.can_deal_board():
        return False
    return True


def payoff(state, player: int) -> float:
    """calculates win/loss"""
    return (float(state.stacks[player]) - BUY_IN) / BUY_IN


def _deal_board(state):
    """deals board if needed"""
    for _ in range(4):
        if not (hasattr(state, "can_deal_board") and state.can_deal_board()):
            break
        try:
            dealable = list(state.get_dealable_cards())
            n        = state.board_dealing_count
            cards    = random.sample(dealable, n)
            state.deal_board("".join(c.rank + c.suit for c in cards))
        except Exception:
            break


def apply_action(state, action: int, player: int) -> bool:
    """executes an action in pokerkit"""
    try:
        if action == FOLD:
            state.fold()
        elif action in (CHECK, CALL):
            state.check_or_call()
        elif action in (RAISE_QUARTER, RAISE_HALF, RAISE_POT):
            pot     = float(state.total_pot_amount)
            stack   = float(state.stacks[player])
            mult    = {RAISE_QUARTER: 0.25, RAISE_HALF: 0.50, RAISE_POT: 1.00}[action]
            cur_bet = float(state.bets[player]) if player < len(state.bets) else 0.0
            target  = min(cur_bet + pot * mult, stack + cur_bet)
            min_r   = state.min_completion_betting_or_raising_to_amount
            if min_r is not None:
                target = max(target, float(min_r))
            state.complete_bet_or_raise_to(target)
        _deal_board(state)
        return True
    except Exception:
        if action != FOLD:
            try:
                state.check_or_call()
                _deal_board(state)
                return True
            except Exception:
                pass
        return False


def legal_mask(state, player: int) -> list:
    """returns a binary mask of legal actions"""
    mask = [0] * NUM_ACTIONS
    if is_terminal(state):
        mask[CHECK] = 1
        return mask
    mask[FOLD] = 1
    try:
        if state.can_check_or_call():
            bets = list(state.bets)
            my   = float(bets[player])     if bets          else 0.0
            opp  = float(bets[1 - player]) if len(bets) > 1 else 0.0
            if opp > my:
                mask[CALL]  = 1
            else:
                mask[CHECK] = 1
                mask[FOLD]  = 0
        if state.can_complete_bet_or_raise_to():
            mask[RAISE_QUARTER] = mask[RAISE_HALF] = mask[RAISE_POT] = 1
    except Exception:
        mask[CHECK] = 1
        mask[FOLD]  = 0
    return mask



def regret_match(raw: np.ndarray, legal: list) -> np.ndarray:
    """perform regret matching"""
    strategy = np.zeros(NUM_ACTIONS, dtype=np.float32)
    pos      = np.array([max(0.0, float(raw[a])) for a in legal], dtype=np.float32)
    total    = pos.sum()
    if total > 0:
        for i, a in enumerate(legal):
            strategy[a] = pos[i] / total
    else:
        for a in legal:
            strategy[a] = 1.0 / len(legal)
    return strategy



class RegretNet(nn.Module):
    
    def __init__(self, obs_dim: int = OBS_DIM, hidden: int = 256, n_actions: int = NUM_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden,  hidden), nn.ReLU(),
            nn.Linear(hidden,  n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StrategyNet(nn.Module):

    def __init__(self, obs_dim: int = OBS_DIM, hidden: int = 256, n_actions: int = NUM_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden,  hidden), nn.ReLU(),
            nn.Linear(hidden,  n_actions),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



class ReservoirBuffer:

    def __init__(self, max_size: int = 500_000):
        self.max_size  = max_size
        self._obs:     list = []
        self._targets: list = []
        self._weights: list = []
        self._n_added  = 0

    def add(self, obs: np.ndarray, target: np.ndarray, weight: float = 1.0):
        if len(self._obs) < self.max_size:
            self._obs.append(obs)
            self._targets.append(target)
            self._weights.append(weight)
        else:
            idx = random.randint(0, self._n_added)
            if idx < self.max_size:
                self._obs[idx]     = obs
                self._targets[idx] = target
                self._weights[idx] = weight
        self._n_added += 1

    def sample_batch(self, batch_size: int, device):
        n = len(self._obs)
        if n == 0:
            return None
        idx     = random.sample(range(n), min(batch_size, n))
        obs     = torch.tensor(np.stack([self._obs[i]     for i in idx]),
                               dtype=torch.float32, device=device)
        targets = torch.tensor(np.stack([self._targets[i] for i in idx]),
                               dtype=torch.float32, device=device)
        weights = torch.tensor([self._weights[i] for i in idx],
                               dtype=torch.float32, device=device)
        return obs, targets, weights

    def __len__(self) -> int:
        return len(self._obs)



def traverse(
    state,
    action_history: list,
    training_player: int,
    regret_nets: list,        
    strategy_buffers: list,   
    regret_buffers: list,     
    iteration: int,
    device,
) -> float:
    
    if is_terminal(state):
        return payoff(state, training_player)

    actor = state.actor_index
    mask  = legal_mask(state, actor)
    legal = [i for i, v in enumerate(mask) if v]

    if not legal:
        return payoff(state, training_player)

    obs   = build_obs(state, actor, action_history)
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        raw_regrets = regret_nets[actor](obs_t).squeeze(0).cpu().numpy()

    strategy = regret_match(raw_regrets, legal)

    if actor == training_player:
        action_values = np.zeros(NUM_ACTIONS, dtype=np.float32)
        for a in legal:
            s_copy = copy.deepcopy(state)
            apply_action(s_copy, a, actor)
            action_values[a] = traverse(
                s_copy, action_history + [(actor, a)],
                training_player, regret_nets,
                strategy_buffers, regret_buffers,
                iteration, device,
            )

        value           = float(np.dot(strategy, action_values))
        instant_regrets = action_values - value   
        regret_buffers[training_player].add(obs, instant_regrets)
        return value

    else:
        strategy_buffers[actor].add(obs, strategy, weight=float(iteration + 1))

        probs  = np.array([strategy[a] for a in legal], dtype=float)
        probs /= probs.sum()
        chosen = int(np.random.choice(legal, p=probs))

        apply_action(state, chosen, actor)
        return traverse(
            state, action_history + [(actor, chosen)],
            training_player, regret_nets,
            strategy_buffers, regret_buffers,
            iteration, device,
        )



def train_network(
    net: nn.Module,
    optimizer,
    buffer: ReservoirBuffer,
    batch_size: int,
    n_steps: int,
    device,
) -> float:
    """train Deep CFR model"""
    if len(buffer) < batch_size:
        return 0.0
    net.train()
    total_loss = 0.0
    for _ in range(n_steps):
        result = buffer.sample_batch(batch_size, device)
        if result is None:
            break
        obs, targets, weights = result
        preds = net(obs)
        loss  = (weights.unsqueeze(1) * (preds - targets) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    net.eval()
    return total_loss / max(n_steps, 1)



def _save_model(regret_nets, strategy_nets, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(
        {
            "regret_net_0":   regret_nets[0].state_dict(),
            "regret_net_1":   regret_nets[1].state_dict(),
            "strategy_net_0": strategy_nets[0].state_dict(),
            "strategy_net_1": strategy_nets[1].state_dict(),
        },
        path,
    )



def train(
    n_iterations:  int = 1_000,
    traversals:    int = 100,
    batch_size:    int = 512,
    train_steps:   int = 200,
    output_path:   str = None,
):
    """main training loop
    """
    if output_path is None:
        output_path = os.path.join(SCRIPT_DIR, "cfr_model.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    regret_nets   = [RegretNet().to(device).eval()   for _ in range(2)]
    strategy_nets = [StrategyNet().to(device).eval() for _ in range(2)]
    regret_opts   = [Adam(regret_nets[p].parameters(),   lr=1e-3) for p in range(2)]
    strategy_opts = [Adam(strategy_nets[p].parameters(), lr=1e-3) for p in range(2)]

    regret_bufs   = [ReservoirBuffer(max_size=500_000)   for _ in range(2)]
    strategy_bufs = [ReservoirBuffer(max_size=1_000_000) for _ in range(2)]

    print("=" * 70)
    print("  Deep CFR — Heads-Up No-Limit Texas Hold'em")
    print(f"  Device             : {device}")
    print(f"  Observation dim    : {OBS_DIM}")
    print(f"  Iterations         : {n_iterations:,}")
    print(f"  Traversals/iter    : {traversals}  (×2 players = {2*traversals} deals/iter)")
    print(f"  Batch size         : {batch_size}")
    print(f"  Train steps/iter   : {train_steps}")
    print(f"  Output             : {output_path}")
    print("=" * 70)

    for t in range(n_iterations):
        for _ in range(traversals):
            deck = DECK[:]
            random.shuffle(deck)
            hole0, hole1 = deck[0] + deck[1], deck[2] + deck[3]
            for player in (0, 1):
                state = create_state(hole0, hole1)
                traverse(
                    state, [], player,
                    regret_nets, strategy_bufs, regret_bufs,
                    t, device,
                )

        for p in (0, 1):
            train_network(regret_nets[p],   regret_opts[p],   regret_bufs[p],
                          batch_size, train_steps, device)
            train_network(strategy_nets[p], strategy_opts[p], strategy_bufs[p],
                          batch_size, train_steps, device)

        if (t + 1) % 50 == 0:
            print(
                f"  iter {t+1:>5} | "
                f"reg0: {len(regret_bufs[0]):>7,} | "
                f"reg1: {len(regret_bufs[1]):>7,} | "
                f"str0: {len(strategy_bufs[0]):>7,} | "
                f"str1: {len(strategy_bufs[1]):>7,}"
            )

        if (t + 1) % 200 == 0:
            ckpt = output_path.replace(".pt", f"_ckpt{t+1}.pt")
            _save_model(regret_nets, strategy_nets, ckpt)
            print(f"  [checkpoint] → {ckpt}")

    _save_model(regret_nets, strategy_nets, output_path)
    print(f"\nDone.  Models saved → {output_path}")
    return regret_nets, strategy_nets



def query_strategy_net(
    strategy_net: StrategyNet,
    state,
    player: int,
    action_history: list,
    mask: list,
    device,
    deterministic: bool = False,
) -> int:
    
    legal = [i for i, v in enumerate(mask) if v]
    if not legal:
        return CHECK

    obs   = build_obs(state, player, action_history)
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        probs_all = strategy_net(obs_t).squeeze(0).cpu().numpy()

    probs = np.array([probs_all[a] for a in legal], dtype=float)
    probs /= probs.sum()

    if deterministic:
        return legal[int(np.argmax(probs))]
    return int(np.random.choice(legal, p=probs))




if __name__ == "__main__":
    sys.setrecursionlimit(5000)

    parser = argparse.ArgumentParser(
        description="Train a Deep CFR poker agent and save RegretNets + StrategyNets."
    )
    parser.add_argument(
        "--iterations", type=int, default=1_000,
        help="Number of CFR iterations (default: 1 000).",
    )
    parser.add_argument(
        "--traversals", type=int, default=100,
        help="External-sampling traversals per iteration per player (default: 100).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=512,
        help="Mini-batch size for network regression (default: 512).",
    )
    parser.add_argument(
        "--train-steps", type=int, default=200,
        help="Gradient steps applied per network per iteration (default: 200).",
    )
    parser.add_argument(
        "--output", type=str,
        default=os.path.join(SCRIPT_DIR, "cfr_model.pt"),
        help="Output .pt file path (default: CFR_test/cfr_model.pt).",
    )
    args = parser.parse_args()

    train(
        n_iterations = args.iterations,
        traversals   = args.traversals,
        batch_size   = args.batch_size,
        train_steps  = args.train_steps,
        output_path  = args.output,
    )
