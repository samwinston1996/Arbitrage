import threading
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from env.trading_env import TradingEnv
import logging
import os

# Setup logging to track the trade decisions and rewards
logging.basicConfig(level=logging.INFO)

# Function to train the agent on a given timeframe in a separate thread
def train_on_timeframe(timeframe):
    logging.info(f"Training on timeframe: {timeframe}")
    
    # Initialize the custom environment for each timeframe
    env = TradingEnv(timeframe=timeframe)

    model_path = f'models/agent_model_{timeframe}'
    
    # Check if a saved model exists, load it if available
    if os.path.exists(f'{model_path}.zip'):
        model = PPO.load(model_path, env=env)
        logging.info(f"Loaded saved model for M{timeframe}.")
    else:
        # Initialize RL agent using PPO if no saved model exists
        model = PPO('MlpPolicy', env, verbose=1)
        logging.info(f"Training new model for M{timeframe}.")

    # Train the agent for 50,000 timesteps on the given timeframe
    model.learn(total_timesteps=50000)

    # Track rewards for this timeframe
    rewards = []

    # Test the agent and collect rewards
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        rewards.append(reward)  # Collect the reward at each step

        # Log reward every 1000 steps
        if i % 1000 == 0:
            logging.info(f"Timeframe: {timeframe}, Step: {i}, Reward: {reward}")

        if done:
            obs = env.reset()

    # Store the rewards for this timeframe
    timeframe_rewards[timeframe] = rewards

    # Save the trained model for this timeframe
    model.save(model_path)

    # Close the environment
    env.close()

# List of timeframes to test
timeframes = ['M1', 'M2', 'M5', 'M15', 'M30']
timeframe_rewards = {}

# Create and start threads for each timeframe
threads = []
for timeframe in timeframes:
    thread = threading.Thread(target=train_on_timeframe, args=(timeframe,))
    threads.append(thread)
    thread.start()

# Wait for all threads to finish
for thread in threads:
    thread.join()

# Visualize the rewards for each timeframe
plt.figure(figsize=(10, 6))
for timeframe, rewards in timeframe_rewards.items():
    plt.plot(rewards, label=f'Timeframe {timeframe}')
plt.xlabel('Step')
plt.ylabel('Reward')
plt.title('Reward over Time for Different Timeframes')
plt.legend()
plt.show()

logging.info("Training and evaluation complete for all timeframes.")
