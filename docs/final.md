---
layout: default
title: Final Report
---
# Video
<iframe width="560" height="315" src="https://www.youtube.com/embed/bteX7ahuExg?si=uA4iNr1E2_akPVAF" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>


# Project Summary
We chose poker as the focus for this project because it’s a fun, familiar, and surprisingly deep game that naturally captures a lot of interesting decision-making challenges. Most people already have some general intuition about poker — betting, folding, or bluffing — so it’s easy to understand what the agent is trying to do and why its behavior matters. At the same time, poker isn’t just a simple game of luck or fixed rules; it’s a competitive setting where strategy, adaptation, and analysis all play a role. That made it a really engaging domain to explore with how intelligent agents can learn and improve over time.


Poker is definitely not a trivial problem to solve. One of the biggest challenges is that it’s an imperfect information game, meaning players don’t have access to all the relevant information (like the opponent’s cards). On top of that, there’s randomness from the card draws, a huge number of possible game states, and the need to think ahead across multiple rounds of betting. A good poker strategy also involves unpredictability to avoid being exploited, so the agent can’t just follow a fixed set of rules. All of this makes it really hard to design a strong strategy by hand, since the “best” decision often depends on hidden information and long-term outcomes rather than immediate results. Additionally, poker naturally fits into a state → action → reward framework: the agent observes the current game state (cards, bets, position), takes an action (fold, call, raise), and eventually receives a reward based on the outcome of the hand. This structure makes it especially well suited for reinforcement learning, where the agent can learn effective strategies over time by optimizing for long-term expected reward.


This is where machine learning, especially reinforcement learning, becomes a great fit. Instead of trying to hard-code a strategy, we can let the agent learn by playing millions of games and gradually figuring out what works and what doesn’t. Reinforcement learning is designed for exactly this kind of setup, where an agent interacts with an environment, makes decisions, and improves based on rewards over time. In poker, this allows the agent to pick up on patterns like when betting is profitable, how to respond to different opponent behaviors, and even when bluffing makes sense. It also helps handle the complexity of the game by learning general strategies instead of memorizing every possible situation. Overall, ML and RL specifically gave us the tools to tackle a problem that’s too complex and uncertain to solve ourselves and produce strategies that were both  effective and genuinely interesting to analyze.



# Approach

Our approach explores both reinforcement learning and game-theoretic methods for solving heads-up poker. Specifically, we implement three variants:

1. **Masked PPO with historical self-play** — the agent trains against a growing pool of its past policies  
2. **Masked PPO with snapshot-based freezing/unfreezing self-play** — stabilizes learning by periodically fixing opponent policies  
3. **Monte Carlo Counterfactual Regret Minimization (MCCFR)** — a regret-minimization algorithm for imperfect-information games  

These approaches allow us to compare reward-driven learning with methods that explicitly incorporate opponent diversity and game-theoretic reasoning. All methods operate within the same simplified poker environment, enabling direct comparison of their learning dynamics and performance. We evaluate all three approaches in the following sections.

---

# Environment Setup

Our environment is a two-player heads-up No-Limit Texas Hold'em poker game built on a standard 52-card deck.

Each hand proceeds through four betting rounds:
- Preflop  
- Flop  
- Turn  
- River  

Players alternate blinds each hand to avoid positional bias.

At each decision point, the acting player chooses from six discrete actions:
- Fold  
- Check  
- Call  
- Raise ¼ pot  
- Raise ½ pot  
- Raise pot  

This discretized action space keeps the problem tractable while still capturing key strategic tradeoffs.

To reduce complexity, models are trained to play a **single hand (fixed stack size)** rather than tournament-style poker. This avoids complications from varying stack depths and allows faster generalization.

Performance is measured as:
- **Per-hand win rate**, rather than long-term bankroll growth

---

# State, Action, and Reward Representation

## State Representation

At each timestep \( s_t \), the agent observes:

- Private hole cards  
- Visible community cards  
- Pot size and current bet (normalized)  
- Both players’ stack sizes (normalized)  
- Opponent betting history within the hand  

## Action Space

- Fold  
- Check  
- Call  
- Raise ¼ pot  
- Raise ½ pot  
- Raise pot  

## Reward

Rewards are **sparse and delayed**:
- Only given at the end of the hand  
- Equal to normalized net chips won or lost  

---

# Why Masked PPO vs PPO

Poker has **state-dependent legal actions**, so we use **Masked PPO** instead of standard PPO.

- Illegal actions have logits set to \( -\infty \)
- This ensures:
  - Softmax probability = 0 for illegal moves  
  - No gradient flow through invalid actions  

Benefits:
- Prevents illegal moves entirely  
- Improves learning efficiency  
- Reduces noise in gradient updates  

> “Trust region methods are some of the best methods out there” — Roy Fox (2026)

---

# Masked PPO With Historical Self-Play

We train our agent using **Masked Proximal Policy Optimization (PPO)**, an on-policy actor–critic algorithm.

## Training Pipeline

```
Initialize policy πθ
↓
Sample opponent from historical policy pool
↓
Play games to collect trajectories
↓
Store (s, a, logπ, V, r, done)
↓
Compute advantages using GAE
↓
Update policy and value networks with PPO
↓
Add current policy to historical pool
↓
Repeat
```

We collect **2048 timesteps per update**.

Stored tuples:
$$
(s_t, a_t, \log \pi_\theta(a_t | s_t), V_\theta(s_t), r_t, d_t)
$$

## Generalized Advantage Estimation (GAE)

$$
A_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

## PPO Objective

$$
L^{\text{CLIP}}(\theta) =
\mathbb{E}_t \left[
\min\left(
r_t(\theta) A_t,\ 
\text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) A_t
\right)
\right]
$$

## Full Loss Function

$$
L =
L^{\text{CLIP}}
- c_1 \mathbb{E}[(V_\theta(s_t) - R_t)^2]
+ c_2 \mathbb{E}[H(\pi_\theta)]
$$

---

# Why Historical Self-Play

Instead of training against a single evolving opponent:

- Train against a **pool of past policies**
- Reduces overfitting
- Stabilizes learning (less non-stationarity)
- Encourages robustness against diverse strategies

As the pool grows:
- Opponents become stronger and more varied  
- Learned strategies become harder to exploit  

---

# PPO With Fictitious Self-Play (Freezing / Unfreezing)

We also implement **alternating self-play with freezing**:

## Training Process

```
Freeze Agent A → Train Agent B
↓
Freeze Agent B → Train Agent A
↓
Repeat
```

## Why Freezing / Unfreezing

- Eliminates moving-target instability  
- Each agent learns a **best response** to a fixed opponent  
- Produces more stable and interpretable learning  

Over time:
- Both agents iteratively improve  
- Converge toward stronger strategies  

---

# Monte Carlo Counterfactual Regret Minimization (MCCFR)

We implement **MCCFR**, a game-theoretic algorithm for imperfect-information games.

## Regret Definition

$$
R^T(I, a) =
\sum_{t=1}^{T}
\left(
v^\sigma(I, a) - v^\sigma(I)
\right)
$$

## Regret Matching Strategy

$$
\sigma(I, a) =
\frac{\max(R^T(I, a), 0)}
{\sum_{a'} \max(R^T(I, a'), 0)}
$$

## Key Properties

- Minimizes regret at each information set  
- Average strategy converges to a **Nash equilibrium**  
- Produces strategies that are difficult to exploit  

## Implementation Details

- Uses **Monte Carlo sampling** (subset of game tree per iteration)  
- Information sets use:
  - Card strength abstraction  
  - Betting history abstraction  

- Neural networks approximate regret (instead of tabular CFR)

## PPO vs MCCFR

| PPO | MCCFR |
|-----|------|
| Reward-driven | Regret minimization |
| Learns via self-play | Theoretically grounded |
| May be exploitable | Low exploitability |

---

# Hyperparameters

## PPO

| Parameter | Value |
|----------|------|
| Learning rate | 3e-4 |
| Steps per update | 2048 |
| Batch size | 256 |
| Epochs per update | 10 |
| Discount \( \gamma \) | 0.95 |
| GAE \( \lambda \) | 0.9 |
| Clip \( \varepsilon \) | 0.2 |
| Value loss coef | 0.5 |
| Entropy coef | 0.01 |
| Total timesteps | 10,000,000 |
| Snapshot frequency | 50,000 |
| Snapshot pool size | 10 |
| MLP layers | [256, 256] |

- Increased critic capacity: **[256, 256, 256]**
- Evaluation every **5,000 steps** using rolling win rate

## MCCFR

- Up to **100,000 unique hands**
- Tuned abstraction granularity based on convergence

---

# Summary

This framework enables a direct comparison between:
- **Reinforcement learning approaches (PPO variants)**
- **Game-theoretic methods (MCCFR)**

highlighting tradeoffs between:
- empirical performance  
- stability  
- theoretical guarantees  


## Tools Used

## AI Usage
Our primary usage of AI can be broken down into 4 main areas:
1. Highlighting environment inconsistencies during debugging and ensuring proper utilization of CPU during training
2. Providing us with higher level overviews of the models and answering any followup questions we had
3. Giving us baseline hyperparameters with which to iterate from
4. Cleaning up prose and formatting (especially equations) in this document and fixing errors and inconsistencies
