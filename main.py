import gym
from stable_baselines3 import PPO
from env.trading_env import TradingEnv

# Initialize the custom environment
env = TradingEnv()

# Initialize RL agent using PPO
model = PPO('MlpPolicy', env, verbose=1)

# Train the agent
model.learn(total_timesteps=50000)

# Save the trained model
model.save('models/agent_model')

# Close the environment
env.close()

print("Training complete and model saved.")
