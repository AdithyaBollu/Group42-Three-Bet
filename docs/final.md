---
layout: default
title: Final Report
---

# Project Summary
We chose poker as the focus for this project because it’s a fun, familiar, and surprisingly deep game that naturally captures a lot of interesting decision-making challenges. Most people already have some general intuition about poker — betting, folding, or bluffing — so it’s easy to understand what the agent is trying to do and why its behavior matters. At the same time, poker isn’t just a simple game of luck or fixed rules; it’s a competitive setting where strategy, adaptation, and analysis all play a role. That made it a really engaging domain to explore with how intelligent agents can learn and improve over time.


Poker is definitely not a trivial problem to solve. One of the biggest challenges is that it’s an imperfect information game, meaning players don’t have access to all the relevant information (like the opponent’s cards). On top of that, there’s randomness from the card draws, a huge number of possible game states, and the need to think ahead across multiple rounds of betting. A good poker strategy also involves unpredictability to avoid being exploited, so the agent can’t just follow a fixed set of rules. All of this makes it really hard to design a strong strategy by hand, since the “best” decision often depends on hidden information and long-term outcomes rather than immediate results. Additionally, poker naturally fits into a state → action → reward framework: the agent observes the current game state (cards, bets, position), takes an action (fold, call, raise), and eventually receives a reward based on the outcome of the hand. This structure makes it especially well suited for reinforcement learning, where the agent can learn effective strategies over time by optimizing for long-term expected reward.


This is where machine learning, especially reinforcement learning, becomes a great fit. Instead of trying to hard-code a strategy, we can let the agent learn by playing millions of games and gradually figuring out what works and what doesn’t. Reinforcement learning is designed for exactly this kind of setup, where an agent interacts with an environment, makes decisions, and improves based on rewards over time. In poker, this allows the agent to pick up on patterns like when betting is profitable, how to respond to different opponent behaviors, and even when bluffing makes sense. It also helps handle the complexity of the game by learning general strategies instead of memorizing every possible situation. Overall, ML and RL specifically gave us the tools to tackle a problem that’s too complex and uncertain to solve ourselves and produce strategies that were both  effective and genuinely interesting to analyze.



# Approach

## Environment Setup

Our environment is a two-player heads-up No-Limit Texas Hold'em poker game built on a standard 52-card deck. Each hand proceeds through the four standard betting rounds — preflop, flop, turn, and river — with the two players alternating who posts the small and big blind each hand to ensure neither player has a persistent positional advantage. At each decision point, the acting player chooses from six discrete actions: fold, check, raise quarter pot, raise half pot, raise full pot, and all-in. This discretized action space keeps the action space tractable for learning while still capturing the core strategic decisions of standard No-Limit Texas Hold'em.

To limit the size of the state space, both players' stacks reset to a fixed starting amount at the beginning of every hand. This is a deliberate simplification: in a fully-fledged tournament, stack depth varies continuously and dramatically affects optimal strategy — short stacks incentivize shoving, deep stacks enable complex implied-odds play. By resetting stacks each hand, we ensure the agent is always operating in the same effective stack-to-pot ratio regime, making the learning problem stationary and preventing the agent from needing to generalize across wildly different stack dynamics. As a result, our agent is explicitly optimized for a single isolated heads-up hand of poker rather than a full session, meaning the learned strategy reflects standard fixed-stack win-rate rather than long-run bankroll maximization.

## Proximal Policy Optimization With Masking

We train our poker agent using PPO, an on-policy actor–critic algorithm that improves a stochastic policy while constraining update size. This stability is critical in poker due to high reward variance, a non-stationary multi-agent environment, and the need to prevent illegal actions. Our setup pairs PPO with action masking and snapshot-based self-play.

### Method

At each timestep the agent collects a tuple \\(s_t, a_t, \log \pi_\theta(a_t \mid s_t), V_\theta(s_t), r_t, d_t\\), where \\(s_t\\) is the game state, \\(a_t\\) the chosen action, \\(V_\theta(s_t)\\) the critic's value estimate, \\(r_t\\) the reward (chips won/lost), and \\(d_t\\) a done flag. Experience is collected for 2048 steps before each update.

Because poker has state-dependent legal actions (fold, check, call, raise ¼/½/full pot, all-in), we apply a binary action mask \\(m_i \in \{0,1\}\\) and renormalize the policy:

$$\tilde{\pi}(a \mid s) = \frac{\pi(a \mid s) \cdot m_a}{\sum_j \pi(j \mid s) \cdot m_j}$$

This eliminates illegal moves and prevents the policy from wasting gradient updates on impossible actions.

PPO updates the policy by maximizing the clipped surrogate objective:

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta)\, A_t,\ \text{clip}(r_t(\theta),\ 1-\varepsilon,\ 1+\varepsilon)\, A_t \right) \right]$$

where \\(r_t(\theta) = \pi_\theta(a_t \mid s_t) / \pi_{\theta_{\text{old}}}(a_t \mid s_t)\\) is the probability ratio and \\(\varepsilon = 0.2\\) limits destructive updates. Advantages \\(A_t\\) are estimated using Generalized Advantage Estimation (GAE):

$$A_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}, \qquad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

The full training loss combines the clipped policy objective, a value function loss, and an entropy bonus to encourage exploration:

$$L = L^{\text{CLIP}} - c_1 \, \mathbb{E}_t\left[(V_\theta(s_t) - R_t)^2\right] + c_2 \, \mathbb{E}_t\left[H(\pi_\theta(\cdot \mid s_t))\right]$$

where \\(R_t = A_t + V(s_t)\\) are the bootstrapped returns. To better handle poker's high reward variance, we use a larger critic network (layers: 256–256–256) than the actor (128–128).

To stabilize the non-stationary multi-agent setting, we use snapshot self-play: every 50,000 steps the current agent is saved, and the opponent is randomly sampled from previous snapshots, refreshing every 50k steps. This approximates fictitious self-play and creates a curriculum of progressively stronger opponents. We've trained for a total of 10,000,000 environment steps and evaluate every 5,000 steps using a rolling win rate.

### Hyperparameters

Values initially followed recommended defaults suggested by Claude AI; the larger critic architecture and snapshot pool size were tuned manually based on observed training stability.

| Parameter | Value |
|---|---|
| Learning rate | 3e-4 |
| Steps per update (\\(n\\)) | 2048 |
| Batch size | 256 |
| Epochs per update | 10 |
| Discount \\(\gamma\\) | 0.95 |
| GAE \\(\lambda\\) | 0.9 |
| Clip \\(\varepsilon\\) | 0.2 |
| Value loss coef \\(c_1\\) | 0.5 |
| Entropy coef \\(c_2\\) | 0.01 |
| Total timesteps | 10,000,000 |
| Snapshot frequency | 50,000 steps |
| Snapshot pool size | 10 |

## Monte Carlo Counterfactual Regret Minimization (MCCFR)

We also plan to train a poker agent using Monte Carlo Counterfactual Regret Minimization (MCCFR), a game-theoretic algorithm designed specifically for large imperfect-information games. Unlike PPO, which learns via reward maximization, MCCFR iteratively minimizes regret — the difference between the reward actually obtained and the reward that could have been obtained by playing the best action in hindsight. At Nash equilibrium, regret approaches zero for all players. While we have not yet completed a successful training run, we understand the algorithm and outline our planned implementation below.

### Method

MCCFR operates over the full game tree. At each information set \\(I\\) (the set of states indistinguishable to a player given their observations), the algorithm maintains a regret table and a strategy table. The instantaneous regret for action \\(a\\) at information set \\(I\\) on iteration \\(t\\) is:

$$r^t(I, a) = v^\sigma(I, a) - v^\sigma(I)$$

where \\(v^\sigma(I, a)\\) is the counterfactual value of taking action \\(a\\) at \\(I\\) and then following strategy \\(\sigma\\), and \\(v^\sigma(I)\\) is the counterfactual value of the current mixed strategy at \\(I\\). Cumulative regret is then:

$$R^T(I, a) = \sum_{t=1}^{T} r^t(I, a)$$

The strategy at each iteration is updated via regret matching:

$$\sigma^{t+1}(I, a) = \frac{\max(R^T(I, a),\ 0)}{\sum_{a'} \max(R^T(I, a'),\ 0)}$$

The final policy is the average strategy \\(\bar{\sigma}\\) accumulated across all iterations, which converges to a Nash equilibrium as \\(T \to \infty\\). The "Monte Carlo" component refers to sampling only a subset of the game tree per iteration rather than traversing it fully, making it tractable for large games like poker.

In our setting, states \\(s_t\\) encode the player's hole cards, community cards, pot size, stack sizes, and betting history. The action space mirrors our PPO setup (fold, check, call, raise ¼/½/full pot, all-in), with illegal actions excluded via the information set definition rather than a separate mask. Rewards are the chip delta at showdown, normalized by the starting stack.

### Planned Implementation & Hyperparameters

We plan to implement external sampling MCCFR, where on each iteration one player's actions are sampled stochastically and the opponent's subtree is traversed fully, as this variant has favorable convergence properties for two-player zero-sum games. Regret and strategy tables will be stored as dictionaries keyed by information set, using abstracted representations for cards and bet sizes to keep the state space tractable.

Default hyperparameter values were initially sourced from Claude AI, but we plan to tune iteration count and abstraction granularity based on convergence of exploitability.

| Parameter | Planned Value | Tuning Plan |
|---|---|---|
| CFR iterations | 1,000,000 | Tune based on exploitability convergence |
| Sampling strategy | External sampling | Fixed (best suited for 2-player) |
| Card abstraction buckets | 50 | Tune; coarser = faster but weaker |
| Bet abstraction levels | 4 (¼, ½, full, all-in) | Fixed to match PPO action space |
| Regret floor | 0 (vanilla regret matching) | May test regret matching+ |
| Strategy averaging | Full average | May test linear averaging |
| Exploitability eval frequency | Every 10,000 iterations | Fixed |

## Current Status Evaluation

### Snapshot Self-Play Evaluation
<img width="1400" height="400" src="/Group42-Three-Bet/assets/Snapshot_Evaluations.png" />

Looking at our graphs we can see that the reward mean shows relatively small fluctuations across training, with the smoothed value sitting at approximately 0.05 — slightly positive but close to zero. This suggests the agent is marginally above a neutral reward baseline, though there is still room for improvement in terms of earning consistently higher rewards. The value loss shows a steady decrease from ~0.65 to ~0.45, indicating the critic is becoming better calibrated at estimating future returns, which is a positive sign that learning is progressing and a strategy is being devised.

In our evaluation, we ran 10,000 hands between our most recent model (Model A) and a previous snapshot (Model B). 

Model A achieved a lower overall win rate, but demonstrated a significantly higher reward per hand — finishing profitable. This indicates that Model A has learned a more selective strategy, choosing to fold unfavorable hands rather than playing every opportunity. This is a hallmark of sound poker/card game strategy, where knowing when not to play is just as important as knowing when to play.

Model B, by contrast, won more hands in total but generated a negative reward per hand, ultimately losing money overall. This suggests Model B developed an aggressive, indiscriminate strategy — playing most or all hands regardless of their strength. While this inflates the win count, it leads to poor expected value over time, as the losses on weak hands outweigh the gains.

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


## Tools Used

## AI Usage
Our primary usage of AI can be broken down into 4 main areas:
1. Highlighting environment inconsistencies during debugging and ensuring proper utilization of CPU during training
2. Providing us with higher level overviews of the models and answering any followup questions we had
3. Giving us baseline hyperparameters with which to iterate from
4. Cleaning up prose and formatting (especially equations) in this document and fixing errors and inconsistencies
