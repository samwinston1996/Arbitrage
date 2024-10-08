import gym
from stable_baselines3 import PPO

def train_agent(env):
    # Initialize PPO Agent
    model = PPO('MlpPolicy', env, verbose=1)
    
    # Train agent for 100k steps
    model.learn(total_timesteps=100000)
    
    # Save the model
    model.save('models/agent_model')
    
    return model

def load_agent():
    model = PPO.load('models/agent_model')
    return model
