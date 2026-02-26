---
layout: default
title: Status
---

## Video Status Update

<iframe width="560" height="315" src="https://www.youtube.com/embed/k85MC-JA3IM?si=YtuGX8xjCXsMSBd0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

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

### Quantitative
<img src="https://github.com/AdithyaBollu/Group42-Three-Bet/blob/2be5e3d5462332db534f4fc6e00534f855ba1858/graphs.png" alt="Training Graphs">

Looking at our graphs, we see that the approx KL and clip fraction are coming down, which means the policy is stabilizing as there are less "big updates" to the policy. Additionally, this is also visible through the clip fraction as less clips indicate that the model is stabilizing. However, the train entropy loss is starting to increase, which indicates that the model is becoming more deterministic rather than completely stochastic.

### Qualitative
Early in training, the agent exhibited a clear bias toward calling almost every action regardless of hand strength or board texture, suggesting it had not yet learned to differentiate between strong and weak holdings. As training progressed, we began observing more appropriate behavior: the agent started checking back hands on dry, uncoordinated boards rather than blindly continuation betting, and demonstrated a greater willingness to call down with medium-strength holdings against potential bluffs. While the agent is still far from optimal and exhibits exploitable tendencies, the shift from undifferentiated calling to something resembling genuine decision making represents a meaningful improvement and suggests the policy is beginning to capture some of the underlying structure of poker strategy.

## Remaining Goals and Roadmap

Our current prototype is limited in two significant ways: the MCCFR agent has not yet produced a converged strategy, and the two agents exist entirely independently with no mechanism for combining them. Our goals for the remainder of the quarter are therefore to first get MCCFR fully trained and validated via exploitability metrics, then investigate hybrid approaches that merge the two agents' action distributions with the goal of producing a bot that outperforms either agent on its own. We also intend to perform more rigorous comparative evaluation between PPO, MCCFR, and the hybrid, which our current results do not yet support. Finally, time permitting, we would like to wrap the finished agent in a playable GUI with a real-time coaching layer that draws on the agent's value estimates to offer feedback on human decisions, potentially via an open-source LLM plugin.

The challenges we anticipate are substantial. Getting MCCFR to converge in a full poker game tree — even with abstraction — is computationally expensive, and information set explosion remains a real risk of becoming a roadblocking obstacle; we plan to address this through aggressive card and bet abstraction and may fall back to a smaller game variant if necessary. The hybrid weighting experiments also introduce a search problem with no obvious conclusion beyond head-to-head win rate, which is not entirely reliable in poker. We hope to mitigate this by using exploitability as a secondary metric where possible. Finally, building a functional GUI and coaching interface within the quarter is ambitious, and if time is short we will prioritize agent quality and evaluation over the interface, treating the GUI as a stretch goal rather than a core deliverable.

## AI Usage
Our primary usage of AI can be broken down into 4 main areas:
1. Highlighting environment inconsistencies during debugging and ensuring proper utilization of CPU during training
2. Providing us with higher level overviews of the models and answering any followup questions we had
3. Giving us baseline hyperparameters with which to iterate from
4. Cleaning up prose and formatting (especially equations) in this document and fixing errors and inconsistencies
