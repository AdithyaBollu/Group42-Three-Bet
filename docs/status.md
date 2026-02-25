---
layout: default
title: Status
---

## Video Status Update


## Proximal Policy Optimization With Masking

We train our poker agent using PPO, an on-policy actor–critic algorithm that improves a stochastic policy while constraining update size. This stability is critical in poker due to high reward variance, a non-stationary multi-agent environment, and the need to prevent illegal actions. Our setup pairs PPO with action masking and snapshot-based self-play.

### Method

At each timestep the agent collects a tuple $(s_t, a_t, \log \pi_\theta(a_t \mid s_t), V_\theta(s_t), r_t, d_t)$, where $s_t$ is the game state, $a_t$ the chosen action, $V_\theta(s_t)$ the critic's value estimate, $r_t$ the reward (chips won/lost), and $d_t$ a done flag. Experience is collected for 2048 steps before each update.

Because poker has state-dependent legal actions (fold, check, call, raise ¼/½/full pot, all-in), we apply a binary action mask $m_i \in \{0,1\}$ and renormalize the policy:

$$\tilde{\pi}(a \mid s) = \frac{\pi(a \mid s) \cdot m_a}{\sum_j \pi(j \mid s) \cdot m_j}$$

This eliminates illegal moves and prevents the policy from wasting gradient updates on impossible actions.

PPO updates the policy by maximizing the clipped surrogate objective:

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta)\, A_t,\ \text{clip}(r_t(\theta),\ 1-\varepsilon,\ 1+\varepsilon)\, A_t \right) \right]$$

where $r_t(\theta) = \pi_\theta(a_t \mid s_t) / \pi_{\theta_{\text{old}}}(a_t \mid s_t)$ is the probability ratio and $\varepsilon = 0.2$ limits destructive updates. Advantages $A_t$ are estimated using Generalized Advantage Estimation (GAE):

$$A_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}, \qquad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

The full training loss combines the clipped policy objective, a value function loss, and an entropy bonus to encourage exploration:

$$L = L^{\text{CLIP}} - c_1 \, \mathbb{E}_t\left[(V_\theta(s_t) - R_t)^2\right] + c_2 \, \mathbb{E}_t\left[H(\pi_\theta(\cdot \mid s_t))\right]$$

where $R_t = A_t + V(s_t)$ are the bootstrapped returns. To better handle poker's high reward variance, we use a larger critic network (layers: 256–256–256) than the actor (128–128).

To stabilize the non-stationary multi-agent setting, we use **snapshot self-play**: every 50,000 steps the current agent is saved, and the opponent is sampled from the 10 most recent snapshots, 
refreshing every 300 episodes. This approximates fictitious self-play and creates a curriculum of progressively stronger opponents. 
We've trained it for a total of 10,000,000 environment steps and evaluate every 10,000 steps over 200 games using rolling win rate.

### Hyperparameters

Values initially followed recommended defaults suggested by Claude AI; the larger critic architecture and snapshot pool size were tuned manually based on observed training stability.

| Parameter | Value |
|---|---|
| Learning rate | 3e-4 |
| Steps per update ($n$) | 2048 |
| Batch size | 256 |
| Epochs per update | 10 |
| Discount $\gamma$ | 0.95 |
| GAE $\lambda$ | 0.9 |
| Clip $\varepsilon$ | 0.2 |
| Value loss coef $c_1$ | 0.5 |
| Entropy coef $c_2$ | 0.01 |
| Total timesteps | 10,000,000 |
| Snapshot frequency | 50,000 steps |
| Snapshot pool size | 10 |


## Monte Carlo Counterfactual Regret Minimization (MCCFR)

We also plan to train a poker agent using Monte Carlo Counterfactual Regret Minimization (MCCFR), a game-theoretic algorithm designed 
specifically for large imperfect-information games. Unlike PPO, which learns via reward maximization, MCCFR iteratively minimizes *regret* — 
the difference between the reward actually obtained and the reward that could have been obtained by playing the best action in hindsight. 
At Nash equilibrium, regret approaches zero for all players. While we have not yet completed a successful training run, we understand the 
algorithm and outline our planned implementation below.

### Method

MCCFR operates over the full game tree. At each information set $I$ (the set of states indistinguishable to a player given their observations), 
the algorithm maintains a **regret table** and a strategy table. The instantaneous regret for action $a$ at information set $I$ on iteration $t$ is:

$$r^t(I, a) = v^\sigma(I, a) - v^\sigma(I)$$

where $v^\sigma(I, a)$ is the counterfactual value of taking action $a$ at $I$ and then following strategy $\sigma$, and 
$v^\sigma(I)$ is the counterfactual value of the current mixed strategy at $I$. Cumulative regret is then:

$$R^T(I, a) = \sum_{t=1}^{T} r^t(I, a)$$

The strategy at each iteration is updated via regret matching:

$$\sigma^{t+1}(I, a) = \frac{\max(R^T(I, a),\ 0)}{\sum_{a'} \max(R^T(I, a'),\ 0)}$$

The final policy is the **average strategy** $\bar{\sigma}$ accumulated across all iterations, which converges to a Nash equilibrium as $T \to \infty$. The "Monte Carlo" component refers to sampling only a subset of the game tree per iteration (outcome sampling or external sampling) rather than traversing it fully, making it tractable for large games like poker [(Lanctot et al., 2009)](https://papers.nips.cc/paper/2009/hash/00411460f7c92d2124a67ea0f4cb5f85-Abstract.html).

In our setting, states $s_t$ encode the player's hole cards, community cards, pot size, stack sizes, and betting history. The action space mirrors our PPO setup (fold, check, call, raise ¼/½/full pot, all-in), with illegal actions excluded via the information set definition rather than a separate mask. Rewards are the chip delta at showdown, normalized by the starting stack.

### Planned Implementation & Hyperparameters

We plan to implement **external sampling MCCFR**, where on each iteration one player's actions are sampled stochastically and the opponent's subtree is traversed fully, as this variant has favorable convergence properties for two-player zero-sum games [(Lanctot et al., 2009)](https://papers.nips.cc/paper/2009/hash/00411460f7c92d2124a67ea0f4cb5f85-Abstract.html). Regret and strategy tables will be stored as dictionaries keyed by information set, using abstracted bucket representations for cards and bet sizes to keep the state space tractable.

Default hyperparameter values follow Lanctot et al. (2009); we plan to tune iteration count and abstraction granularity based on convergence of exploitability.

| Parameter | Planned Value | Tuning Plan |
|---|---|---|
| CFR iterations | 1,000,000 | Tune based on exploitability convergence |
| Sampling strategy | External sampling | Fixed (best suited for 2-player) |
| Card abstraction buckets | 50 | Tune; coarser = faster but weaker |
| Bet abstraction levels | 4 (¼, ½, full, all-in) | Fixed to match PPO action space |
| Regret floor | 0 (vanilla regret matching) | May test regret matching+ |
| Strategy averaging | Full average | May test linear averaging |
| Exploitability eval frequency | Every 10,000 iterations | Fixed |

# Current Status Evaluation

## Remaining Goals / Roadmap

Looking ahead, our primary objective is to get the MCCFR agent fully trained and producing a converged, stable strategy. Once both the PPO and MCCFR agents are independently stable, 
we plan to investigate hybrid  approaches that combine their outputs — PPO brings adaptive intuition shaped by its self-play experience, 
while MCCFR contributes its game-theoretic Nash equilibrium reasoning. We intend to experiment with different weighting schemes for merging the two agents' action distributions, 
ranging from fixed convex combinations to learned or state-conditional weightings, with the goal of identifying configurations that outperform either agent in isolation.

Beyond the core agents, we aim to wrap the final bot in a full GUI that allows a human player to sit down and play poker directly against it. This would make the 
project accessible far beyond a research context, turning it into a genuinely playable and demonstrable product. Alongside the gameplay experience, we would like to incorporate 
a real-time advice and coaching layer — drawing on the agent's internal value estimates and probabilities to offer the human player feedback on their decisions, 
such as flagging folds that had positive expected value or highlighting missed bluffing opportunities, most likely via a plugin with an opensource LLM. The combination of a 
strong hybrid agent and an interactive coaching interface would transform the project from a training experiment into a practical tool for understanding and improving poker strategy.

# Resources Used
