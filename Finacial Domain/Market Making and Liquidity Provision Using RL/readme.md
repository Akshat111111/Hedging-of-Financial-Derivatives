# Reinforcement Learning for Market Making and Liquidity Provision

## Overview

This project implements a reinforcement learning (RL) agent for market making and liquidity provision in financial markets. It uses Python, OpenAI Gym, and Stable Baselines3 for training the RL agent.

## Reinforcement Learning Steps

1. **Define RL Environment**: Create a custom Gym environment (`market_making_env.py`) that simulates the market making dynamics, including order book data, trade volumes, and bid-ask spreads.

2. **State Representation**: Represent the state of the environment using relevant features such as bid/ask prices, volumes, market trends, etc.

3. **Reward Function**: Define a reward function that incentivizes the RL agent to maintain liquidity, minimize inventory risk, and earn bid-ask spreads effectively.

4. **Training the RL Agent**: Train the RL agent using Stable Baselines3 (`main.py`). Use algorithms like Proximal Policy Optimization (PPO) to optimize trading strategies based on interactions with the defined environment.

5. **Evaluation**: Evaluate the trained RL agent's performance using metrics such as average reward and standard deviation over multiple episodes (`main.py evaluate`).





