import MetaTrader5 as mt5
import gym
from gym import spaces
import numpy as np
import talib as ta
from stable_baselines3 import RecurrentPPO
import logging
import time
import pandas as pd
import os
from datetime import datetime

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Initialize MetaTrader5
if not mt5.initialize():
    logging.error("MetaTrader5 initialization failed")
    quit()

# Login to MetaTrader5 with credentials
login = 62061021  # Replace with your account number
password = 'your_password'  # Replace with your password
server = 'PepperstoneUK-Demo'  # Replace with your server name

if not mt5.login(login, password=password, server=server):
    logging.error(f"Failed to login to MetaTrader5, error: {mt5.last_error()}")
    mt5.shutdown()
    quit()
else:
    logging.info("MetaTrader5 initialized and logged in")

# Define the symbol filling mode flags
SYMBOL_FILLING_FLAG_FOK = 1 << 0     # 1
SYMBOL_FILLING_FLAG_IOC = 1 << 1     # 2
SYMBOL_FILLING_FLAG_RETURN = 1 << 2  # 4

class TradingEnv(gym.Env):
    def __init__(self, symbol='BTCUSD', timeframe='M1', lot_size=0.01):
        super(TradingEnv, self).__init__()

        # Set symbol and timeframe
        self.symbol = symbol
        self.timeframe = getattr(mt5, f'TIMEFRAME_{timeframe}')
        self.timeframe_name = timeframe
        self.lot_size = lot_size

        # Define action and observation space
        # Actions: 0 = Buy, 1 = Sell, 2 = Hold, 3 = Close Position
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)

        # Initialize variables
        self.prev_observation = None
        self.position = None  # Initialize position to None
        self.market_closed = False  # Initialize market_closed flag
        self.last_log_time = 0  # For controlling log output frequency
        self.current_profit = 0  # To store current profit
        self.position_type = None  # Track the type of the open position

        # Initialize account info
        self.account_info = mt5.account_info()
        if self.account_info is None:
            logging.error("Failed to get account info")
            mt5.shutdown()
            quit()

    def reset(self):
        self.prev_observation = None
        self.position = None  # Reset position to None
        self.market_closed = False  # Reset market_closed flag
        self.last_log_time = 0  # Reset log timer
        self.current_profit = 0  # Reset profit
        self.position_type = None  # Reset position type
        logging.info(f"Environment reset for timeframe {self.timeframe_name}")
        return np.zeros(11, dtype=np.float32)

    def step(self, action):
        # Retrieve market data
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, 500)

        if rates is None or len(rates) == 0:
            logging.error("No market data retrieved")
            time.sleep(60)  # Wait before retrying
            return np.zeros(11), 0, False, {}

        # Prepare data
        close_prices = rates['close'].astype(np.float64)
        high_prices = rates['high'].astype(np.float64)
        low_prices = rates['low'].astype(np.float64)
        volume = rates['tick_volume'].astype(np.float64)

        # Calculate technical indicators
        ema50 = ta.EMA(close_prices, timeperiod=50)[-1]
        rsi = ta.RSI(close_prices, timeperiod=14)[-1]
        adx = ta.ADX(high_prices, low_prices, close_prices, timeperiod=14)[-1]
        volatility = np.std(close_prices[-20:])
        supertrend = self.calculate_supertrend(high_prices, low_prices, close_prices)

        current_price = close_prices[-1]

        # Observation
        obs = np.array([
            current_price,        # Latest close price
            ema50,
            rsi,
            adx,
            supertrend,
            np.min(close_prices[-50:]),
            np.max(close_prices[-50:]),
            np.mean(close_prices[-50:]),
            np.std(close_prices[-50:]),
            volume[-1],
            volatility
        ], dtype=np.float32)

        # Execute action and calculate reward
        reward = self.execute_trade(action, current_price)

        # Log current market price and action every 10 seconds
        current_time = time.time()
        if current_time - self.last_log_time >= 10:
            self.log_market_data(obs, action, reward)
            self.last_log_time = current_time

        # Check if done (we'll keep it always False for live trading)
        done = False

        info = {}

        self.prev_observation = obs

        return obs, reward, done, info

    def calculate_supertrend(self, high, low, close):
        # Implement SuperTrend calculation or use a library function
        # Placeholder implementation
        if close[-1] > ta.SMA(close, timeperiod=10)[-1]:
            return 1  # Bullish
        else:
            return -1  # Bearish

    def execute_trade(self, action, current_price):
        reward = 0

        # Get positions for this symbol and timeframe
        all_positions = mt5.positions_get(symbol=self.symbol)
        positions = [pos for pos in all_positions if pos.comment == f"RL Agent {self.timeframe_name}"]
        positions_count = len(positions)

        if positions_count > 0:
            # There is an open position
            position = positions[0]
            self.position_type = 'buy' if position.type == mt5.ORDER_TYPE_BUY else 'sell'

            # Calculate current profit
            current_profit = self.calculate_current_profit(position)

            # Agent can decide to close the position
            if action == 3:  # Close Position
                result = self.close_position(position)
                if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                    logging.info(f"Closed position at {current_price} with profit {position.profit} ({self.timeframe_name})")
                    reward += current_profit  # Reward is the profit from the trade
                    self.position = None
                    self.position_type = None
            else:
                # Provide continuous reward based on current profit
                reward += current_profit
        else:
            # No open position
            self.position_type = None
            if action == 0:  # Buy
                result = self.open_position('BUY')
                if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                    logging.info(f"Opened Buy position at {result.price} on {self.symbol} ({self.timeframe_name})")
                    self.position = 'buy'
                    self.position_type = 'buy'
            elif action == 1:  # Sell
                result = self.open_position('SELL')
                if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                    logging.info(f"Opened Sell position at {result.price} on {self.symbol} ({self.timeframe_name})")
                    self.position = 'sell'
                    self.position_type = 'sell'
            else:
                # Hold
                self.position = None
                reward += 0  # No reward for holding

        return reward

    def calculate_current_profit(self, position):
        # Determine the position's current profit
        if position.type == mt5.ORDER_TYPE_BUY:
            current_price = mt5.symbol_info_tick(self.symbol).bid
            current_profit = (current_price - position.price_open) * position.volume * mt5.symbol_info(self.symbol).trade_contract_size
        else:  # SELL position
            current_price = mt5.symbol_info_tick(self.symbol).ask
            current_profit = (position.price_open - current_price) * position.volume * mt5.symbol_info(self.symbol).trade_contract_size

        self.current_profit = current_profit
        return current_profit

    def open_position(self, direction):
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            logging.error(f"{self.symbol} not found")
            return None

        if not symbol_info.visible:
            if not mt5.symbol_select(self.symbol, True):
                logging.error(f"Failed to select {self.symbol}")
                return None

        price = mt5.symbol_info_tick(self.symbol).ask if direction == 'BUY' else mt5.symbol_info_tick(self.symbol).bid

        # Determine filling mode
        filling_type = self.get_filling_mode(symbol_info)
        if filling_type is None:
            return None

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": self.lot_size,
            "type": mt5.ORDER_TYPE_BUY if direction == 'BUY' else mt5.ORDER_TYPE_SELL,
            "price": price,
            "deviation": 10,
            "magic": 234000,
            "comment": f"RL Agent {self.timeframe_name}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_type,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Failed to open position: {result.comment}")
            return None

        return result

    def close_position(self, position):
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            logging.error(f"{self.symbol} not found")
            return None

        # Determine filling mode
        filling_type = self.get_filling_mode(symbol_info)
        if filling_type is None:
            return None

        direction = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(self.symbol).bid if direction == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(self.symbol).ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": position.volume,
            "type": direction,
            "position": position.ticket,
            "price": price,
            "deviation": 10,
            "magic": 234000,
            "comment": f"RL Agent Close {self.timeframe_name}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_type,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Failed to close position: {result.comment}")
            return None

        return result

    def get_filling_mode(self, symbol_info):
        filling_modes = []
        if symbol_info.filling_mode & SYMBOL_FILLING_FLAG_FOK:
            filling_modes.append(mt5.ORDER_FILLING_FOK)
        if symbol_info.filling_mode & SYMBOL_FILLING_FLAG_IOC:
            filling_modes.append(mt5.ORDER_FILLING_IOC)
        if symbol_info.filling_mode & SYMBOL_FILLING_FLAG_RETURN:
            filling_modes.append(mt5.ORDER_FILLING_RETURN)
        if filling_modes:
            return filling_modes[0]  # Use the first supported filling mode
        else:
            logging.error(f"No acceptable filling mode found for symbol {self.symbol}")
            return None

    def log_market_data(self, obs, action, reward):
        data = {
            'Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Symbol': self.symbol,
            'Timeframe': self.timeframe_name,
            'Price': obs[0],
            'Action': ['Buy', 'Sell', 'Hold', 'Close'][action],
            'Position': self.position_type,
            'Current Profit': round(self.current_profit, 2),
            'Reward': round(reward, 2)
        }
        # Print as a single line without headers
        output = f"{data['Time']} | {data['Symbol']} | TF: {data['Timeframe']} | Price: {data['Price']:.2f} | Action: {data['Action']} | Position: {data['Position']} | Profit: {data['Current Profit']:.2f} | Reward: {data['Reward']:.2f}"
        print(output)

    def render(self, mode='human'):
        pass

    def close(self):
        pass

# Initialize environments and agents for multiple timeframes
timeframes = ['M1', 'M5', 'M15']
envs = {}
models = {}
model_paths = {}
training_start_time = time.time()
training_duration = 24 * 60 * 60  # Train for 1 day (24 hours)
save_interval = 60 * 60  # Save models every hour
last_save_time = training_start_time

for tf in timeframes:
    envs[tf] = TradingEnv(symbol='BTCUSD', timeframe=tf, lot_size=0.01)
    model_path = f"recurrent_ppo_model_{tf}.zip"
    model_paths[tf] = model_path

    if os.path.exists(model_path):
        # Load the existing model
        models[tf] = RecurrentPPO.load(model_path, env=envs[tf])
        logging.info(f"Loaded existing model for timeframe {tf}")
    else:
        # Create a new model with LSTM policy
        models[tf] = RecurrentPPO('MlpLstmPolicy', envs[tf], verbose=0)
        logging.info(f"Created new RecurrentPPO model for timeframe {tf}")

# Live trading loop
try:
    obs = {tf: envs[tf].reset() for tf in timeframes}
    lstm_states = {tf: None for tf in timeframes}  # Initialize LSTM states
    episode_starts = {tf: True for tf in timeframes}  # Track the start of episodes

    while True:
        current_time = time.time()
        elapsed_time = current_time - training_start_time

        if elapsed_time >= training_duration:
            logging.info("Training duration reached. Saving models and stopping training.")
            # Save the models
            for tf in timeframes:
                models[tf].save(model_paths[tf])
            break  # Exit the loop

        # Save models periodically
        if current_time - last_save_time >= save_interval:
            for tf in timeframes:
                models[tf].save(model_paths[tf])
            last_save_time = current_time
            logging.info("Models saved periodically.")

        for tf in timeframes:
            env = envs[tf]
            # Get the previous LSTM state and episode start
            lstm_state = lstm_states[tf]
            episode_start = episode_starts[tf]

            action, lstm_states[tf] = models[tf].predict(obs[tf], state=lstm_state, episode_start=episode_start, deterministic=False)
            episode_starts[tf] = False  # After the first step, episode has started

            try:
                obs[tf], reward, done, info = env.step(action)
            except Exception as e:
                logging.error(f"Error in timeframe {tf}: {e}")
                # Sleep to prevent flooding and retry after some time
                time.sleep(60)
                continue  # Skip to the next timeframe

            if reward != 0:
                # Train the model with the obtained reward
                # Note: For RecurrentPPO, we typically collect rollouts and train periodically
                pass  # In live trading, immediate training may not be feasible

            # If done, reset the environment and LSTM states
            if done:
                obs[tf] = env.reset()
                lstm_states[tf] = None
                episode_starts[tf] = True

            # Sleep for a short time to limit the rate of requests
            time.sleep(1)

except KeyboardInterrupt:
    logging.info("Interrupted by user")

finally:
    # Save models before exiting
    for tf in timeframes:
        models[tf].save(model_paths[tf])
        envs[tf].close()
    mt5.shutdown()
    logging.info("Trading session ended")
