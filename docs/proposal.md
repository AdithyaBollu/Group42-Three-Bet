---
layout: default
title: Proposal
---


## Summary of the Project
The project is application driven and aims to build a poker bot that can perform well against clones of itself. We want it to perform very well when managing a bankroll and keep a net positive gain. The inputs it receives are the cards that are dealt to it, the cards on the board, and the actions done by clones of itself. The outputs are the decisions the bot makes: check, fold, raise, and call. The objective for the bot is to win.


## Project Goals
**Minimum goal**: To achieve a model that can play basic poker.

**Realistic goal**: To achieve a model that can play poker well and achieve a net positive gain.

**Moonshot goal**: To achieve a smart model that can play poker really well and have a 90% win rate, and possibly win against humans.


## AI/ML Algorithims
- DQN
- Policy gradient methods
  - Actor-Critic or REINFORCE
- Memory models


## Evaluation Plan
Our quantitative evaluation plan will be to see how well the poker bot does against opponents who play randomly, ie. take a random action (fold, call, raise, all-in) every time it is their turn regardless of what cards are present. Eventually the bot will learn what it needs to do in order to win. Later on we will try to see how well it does against itself, and see how many rounds the poker bot can last without losing everything. 

Our qualitative evaluation plan will be to see how well it does on off policy data, such as playing a human. We will also evaluate its play so that it doesn’t try to reward-hack, or do an unexpected behavior due to the nature of the reward function. In particular we can create specific scenarios where we expect the bot to fold, raise, or check and see if the bot aligns with what we expect.


## AI Tool Usage
We plan to use AI tools like Claude and ChatGPT to help us debug our code if we get stuck or help us understand complex concepts.
