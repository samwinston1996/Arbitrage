import gym
from stable_baselines3 import PPO
from env.trading_env import TradingEnv

# Load the environment
env = TradingEnv()

# Load the trained agent
model = PPO.load('models/agent_model')

# Evaluate the agent's performance
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
