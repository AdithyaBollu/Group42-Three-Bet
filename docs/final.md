---
layout: default
title: Final Report
---
# Video
<iframe width="560" height="315" src="https://www.youtube.com/embed/bteX7ahuExg?si=uA4iNr1E2_akPVAF" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>


# Project Summary
We chose poker as the focus for this project because it’s a fun, familiar, and surprisingly deep game that naturally captures a lot of interesting decision-making challenges. Most people already have some general intuition about poker— betting, folding, or bluffing— so it’s easy to understand what the agent is trying to do and why its behavior matters. At the same time, poker isn’t just a simple game of luck or fixed rules; it’s a competitive setting where strategy, adaptation, and analysis all play a role. That made it a really engaging domain to explore with how intelligent agents can learn and improve over time.


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

## Current Status Evaluation

### Snapshot Self-Play Evaluation
<img width="1400" height="400" src="/Group42-Three-Bet/assets/Snapshot_Evaluations.png" />

Looking at our graphs we can see that the reward mean shows relatively small fluctuations across training, with the smoothed value sitting at approximately 0.05, slightly positive but close to zero. This suggests the agent is marginally above a neutral reward baseline, though there is still room for improvement in terms of earning consistently higher rewards. The value loss shows a steady decrease from ~0.65 to ~0.45, indicating the critic is becoming better calibrated at estimating future returns, which is a positive sign that learning is progressing and a strategy is being devised.

In our evaluation, we ran 10,000 hands between our most recent model (Model A) and a previous snapshot (Model B). 

Model A achieved a lower overall win rate, but demonstrated a significantly higher reward per hand, finishing profitable. This indicates that Model A has learned a more selective strategy, choosing to fold unfavorable hands rather than playing every opportunity. This is a hallmark of sound poker/card game strategy, where knowing when not to play is just as important as knowing when to play.

Model B, by contrast, won more hands in total but generated a negative reward per hand, ultimately losing money overall. This suggests Model B developed an aggressive, indiscriminate strategy, playing most or all hands regardless of their strength. While this inflates the win count, it leads to poor expected value over time, as the losses on weak hands outweigh the gains.

In summary, Model A's behavior reflects a more mature and strategically sound policy, prioritizing quality of play over quantity. The improvement in reward per hand over the snapshot is a strong indicator that the model is learning meaningful game strategy rather than simply maximizing hand participation.


### Fictitious Self-Play Evaluation
<img width="1400" height="400" src="/Group42-Three-Bet/assets/Fictitious_Evaluations.png" />

Looking at the updated graphs we can see that the reward mean continues to show small fluctuations across training, remaining tightly bounded between approximately -0.05 and 0.05, with the smoothed value sitting slightly above zero. In the context of a poker environment, this is a strong indication of stability, suggesting the agent is maintaining performance close to a neutral baseline while avoiding large negative outcomes, though there is still room for improvement in achieving more consistently positive rewards. The value loss shows a continued steady decrease from around ~0.7 to ~0.35–0.4, indicating the critic is becoming increasingly well calibrated at estimating future returns, which is a positive sign that learning is stabilizing and a more refined strategy is being developed.

In our evaluation, we ran 10,000 hands between our two self play agents (Agent A and Agent B). 

Both Model A and Model B have very similar metrics in terms of win percentage and average reward, with both agents hovering close to a 50% win rate and near-zero average chips. However, Model A appears to be doing slightly better overall, with a marginally higher average chip gain (+0.02 vs -0.02) and only a small deficit in total wins compared to losses. These near-identical statistics demonstrate how the two agents are converging toward a Nash equilibrium, where neither agent can significantly outperform the other.

Breaking it down further, we can see some positional differences. Model A performs worse as the big blind (44.5% win rate) but compensates with stronger performance as the small blind (52.2%), whereas Model B shows a similar pattern but slightly more polarized, doing better as the small blind (53.0%) and worse as the big blind (46.8%)

Overall, the symmetry in both win rates and rewards across positions suggests that both agents have learned balanced strategies and are effectively countering each other. The slight edge seen in Model A is minimal and likely within the variance expected over 10,000 hands, further supporting the conclusion that both models are approaching equilibrium-like play rather than one clearly dominating the other.



### MC CFR Evaluation
<img width="900" height="500" src="/Group42-Three-Bet/assets/MCCFR_Evalutions.png" />

For our MC CFR evaluation we ran 10,000 hands between our ficitious self-play Agent A and our CFR model.

From the results we can see that MC CFR’s win rate is near 50% (49.5%), but it is consistently losing in terms of average reward against Agent A (-1.65 vs +1.65). This indicates that while MC CFR is able to win a comparable number of hands, the pots it loses tend to be larger, suggesting suboptimal decision-making in higher-stakes situations. This shows that our MC CFR model is still exploitable and not yet fully converged. We need to keep training it for longer ( We were unable to do so due to time constraints).

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
- CS175 Slides (especially for understanding PPO models)
- [PPO Intro Material](https://www.youtube.com/watch?v=5P7I-xPq8u8)
- [MCCFR for Poker](https://www.youtube.com/watch?v=iU14jOue9Dk)
- [Intro to CFR](https://www.youtube.com/watch?v=ygDt_AumPr0)
- Stable Baselines3
- OpenAI Gymnasium
- Poker Kit py.pi
- Claude


## AI Usage
Our primary usage of AI can be broken down into 4 main areas:
1. Highlighting environment inconsistencies during debugging and ensuring proper utilization of CPU during training
2. Providing us with higher level overviews of the models and answering any followup questions we had
3. Giving us baseline hyperparameters with which to iterate from
4. Cleaning up prose and formatting (especially equations) in this document and fixing errors and inconsistencies
