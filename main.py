import MetaTrader5 as mt5
import gym
from gym import spaces
import numpy as np
import talib as ta
import pandas as pd
from collections import deque
from sb3_contrib import RecurrentPPO
import logging
import time
import os
import pickle
from datetime import datetime, timedelta
import glob

# Initialize logging with reduced verbosity
logging.basicConfig(level=logging.WARNING, format='%(asctime)s [%(levelname)s] %(message)s')

# Initialize MetaTrader5
if not mt5.initialize():
    logging.error("MetaTrader5 initialization failed")
    quit()

# Login to MetaTrader5 with credentials
login = 12345678  # Replace with your account number
password = 'your_password'  # Replace with your password
server = 'YourBroker-Server'  # Replace with your server name

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
    def __init__(self, symbol='BTCUSD', timeframes=['M1', 'M5', 'M15'], lot_size=0.01, data_file_prefix='historical_data'):
        super(TradingEnv, self).__init__()

        # Set symbol and timeframes
        self.symbol = symbol
        self.timeframes = timeframes
        self.lot_size = lot_size
        self.data_file_prefix = data_file_prefix

        # Define action and observation space
        # Actions: 0 = Buy, 1 = Sell, 2 = Hold, 3 = Close Position
        self.action_space = spaces.Discrete(4)

        # Initialize variables
        self.prev_observation = None
        self.position = None  # Initialize position to None
        self.market_closed = False  # Initialize market_closed flag
        self.last_log_time = 0  # For controlling log output frequency
        self.current_profit = 0  # To store current profit
        self.position_type = None  # Track the type of the open position
        self.step_count = 0  # For periodic saving
        self.max_data_points = {}  # To store max data points per timeframe

        # Initialize account info
        self.account_info = mt5.account_info()
        if self.account_info is None:
            logging.error("Failed to get account info")
            mt5.shutdown()
            quit()

        # Store the initial account balance
        self.initial_balance = self.account_info.balance

        # Initialize historical data storage
        self.initialize_historical_data()

        # Update observation space size based on features per timeframe
        self.num_features_per_tf = 17  # Existing features per timeframe
        self.num_additional_features = 1  # Account balance
        total_features = (self.num_features_per_tf * len(self.timeframes)) + self.num_additional_features
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(total_features,), dtype=np.float32)

    def calculate_max_data_points(self, timeframe):
        # Calculate the number of data points for 24 hours
        if timeframe == 'M1':
            data_points = 60 * 24  # 1 data point per minute
        elif timeframe == 'M5':
            data_points = 12 * 24  # 1 data point every 5 minutes
        elif timeframe == 'M15':
            data_points = 4 * 24   # 1 data point every 15 minutes
        else:
            # Default to 1,440 data points (assumes M1)
            data_points = 60 * 24
        return data_points

    def initialize_historical_data(self):
        # Initialize historical data for each timeframe
        self.historical_data = {}
        current_date = datetime.now().strftime('%Y%m%d')
        for tf in self.timeframes:
            self.max_data_points[tf] = self.calculate_max_data_points(tf)
            data_file_tf = f"{self.data_file_prefix}_{tf}_{current_date}.pkl"
            if os.path.exists(data_file_tf):
                with open(data_file_tf, 'rb') as f:
                    self.historical_data[tf] = pickle.load(f)
                logging.info(f"Loaded historical data for timeframe {tf} from {data_file_tf}")
            else:
                # Initialize empty deque with maxlen
                self.historical_data[tf] = deque(maxlen=self.max_data_points[tf])
                logging.info(f"Initialized empty historical data for timeframe {tf}")

    def save_historical_data(self):
        # Save historical data to file for each timeframe with date-based filenames
        current_date = datetime.now().strftime('%Y%m%d')
        for tf in self.timeframes:
            data_file_tf = f"{self.data_file_prefix}_{tf}_{current_date}.pkl"
            with open(data_file_tf, 'wb') as f:
                pickle.dump(self.historical_data[tf], f)
            logging.info(f"Saved historical data for timeframe {tf} to {data_file_tf}")

    def delete_old_data(self, days_to_keep=15):
        # Delete historical data files older than days_to_keep days
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        for file in glob.glob(f"{self.data_file_prefix}_*"):
            # Extract date from filename
            date_part = file.split('_')[-1].replace('.pkl', '')
            try:
                file_date = datetime.strptime(date_part, '%Y%m%d')
                if file_date < cutoff_date:
                    os.remove(file)
                    logging.info(f"Deleted old data file: {file}")
            except ValueError:
                # Skip files that don't match the date format
                continue

    def reset(self):
        self.prev_observation = None
        self.position = None  # Reset position to None
        self.market_closed = False  # Reset market_closed flag
        self.last_log_time = 0  # Reset log timer
        self.current_profit = 0  # Reset profit
        self.position_type = None  # Reset position type
        self.step_count = 0  # Reset step count
        # Historical data is already loaded during initialization
        logging.info("Environment reset")
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def step(self, action):
        obs = []
        for tf in self.timeframes:
            timeframe_mt5 = getattr(mt5, f'TIMEFRAME_{tf}')
            # Retrieve market data for the timeframe
            rates = mt5.copy_rates_from_pos(self.symbol, timeframe_mt5, 0, 500)

            if rates is None or len(rates) == 0:
                logging.error(f"No market data retrieved for timeframe {tf}")
                time.sleep(60)  # Wait before retrying
                return np.zeros(self.observation_space.shape), 0, False, {}

            # Prepare data
            close_prices = rates['close'].astype(np.float64)
            high_prices = rates['high'].astype(np.float64)
            low_prices = rates['low'].astype(np.float64)
            volume = rates['tick_volume'].astype(np.float64)
            times = rates['time']

            # Update historical data
            for i in range(len(close_prices)):
                self.historical_data[tf].append({
                    'time': times[i],
                    'close': close_prices[i],
                    'high': high_prices[i],
                    'low': low_prices[i],
                    'volume': volume[i]
                })

            # Convert historical data to DataFrame for processing
            hist_df = pd.DataFrame(list(self.historical_data[tf]))

            # Calculate technical indicators using historical data
            ema50 = ta.EMA(hist_df['close'], timeperiod=50).iloc[-1]
            rsi = ta.RSI(hist_df['close'], timeperiod=14).iloc[-1]
            adx = ta.ADX(hist_df['high'], hist_df['low'], hist_df['close'], timeperiod=14).iloc[-1]
            volatility = hist_df['close'].rolling(window=20).std().iloc[-1]
            supertrend = self.calculate_supertrend(hist_df['high'], hist_df['low'], hist_df['close'])

            current_price = hist_df['close'].iloc[-1]

            # Additional technical indicators
            macd, macd_signal, macd_hist = ta.MACD(hist_df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
            stochastic_k, stochastic_d = ta.STOCH(hist_df['high'], hist_df['low'], hist_df['close'],
                                                  fastk_period=14, slowk_period=3, slowd_period=3)
            atr = ta.ATR(hist_df['high'], hist_df['low'], hist_df['close'], timeperiod=14).iloc[-1]

            # Calculate support and resistance levels
            support, resistance = self.calculate_support_resistance(hist_df)

            # Calculate trade duration
            trade_duration = 0
            if self.position_type is not None:
                # Calculate trade duration in minutes
                position = mt5.positions_get(symbol=self.symbol)[0]
                trade_duration = (datetime.now() - datetime.fromtimestamp(position.time)).total_seconds() / 60.0

            # Append the features to the observation list
            obs.extend([
                current_price,
                ema50,
                rsi,
                adx,
                supertrend,
                hist_df['close'].min(),
                hist_df['close'].max(),
                hist_df['close'].mean(),
                hist_df['close'].std(),
                hist_df['volume'].iloc[-1],
                volatility,
                macd.iloc[-1],
                stochastic_k.iloc[-1],
                atr,
                support,
                resistance,
                trade_duration  # New feature
            ])

            # Save data for analysis
            self.save_analysis_data(tf, hist_df)

        # Include account balance in the observation
        self.account_info = mt5.account_info()  # Update account info
        account_balance = self.account_info.balance
        obs.append(account_balance)

        # Convert obs to numpy array
        obs = np.array(obs, dtype=np.float32)

        # Get positions for this symbol
        positions = mt5.positions_get(symbol=self.symbol)
        positions_count = len(positions)

        # Modify action based on whether a position is open
        if positions_count > 0:
            # Position is open; limit actions to "Close" or "Hold"
            if action in [0, 1]:
                # Ignore Buy or Sell actions when position is open
                action = 2  # Set action to Hold
            # Execute action and calculate reward
            reward = self.execute_trade(action)
        else:
            # No position is open; actions "Buy" or "Sell" are valid
            if action == 3:
                # Cannot close a position when none is open
                action = 2  # Set action to Hold
            # Execute action and calculate reward
            reward = self.execute_trade(action)

        # Check if done (we'll keep it always False for live trading)
        done = False
        info = {}
        self.prev_observation = obs

        # Increment step count and save historical data periodically
        self.step_count += 1
        if self.step_count % 10 == 0:
            self.save_historical_data()
            self.delete_old_data(days_to_keep=15)

        return obs, reward, done, info

    def save_analysis_data(self, timeframe, hist_df):
        # Save analysis data with date-based filenames
        current_date = datetime.now().strftime('%Y%m%d')
        data_file = f"analysis_data_{timeframe}_{current_date}.csv"
        hist_df['timeframe'] = timeframe  # Add timeframe column
        if not os.path.exists(data_file):
            hist_df.to_csv(data_file, index=False)
        else:
            hist_df.to_csv(data_file, mode='a', header=False, index=False)

    def calculate_support_resistance(self, hist_df):
        # Simple support and resistance calculation using pivots
        pivot = (hist_df['high'].iloc[-1] + hist_df['low'].iloc[-1] + hist_df['close'].iloc[-1]) / 3
        support = (2 * pivot) - hist_df['high'].iloc[-1]
        resistance = (2 * pivot) - hist_df['low'].iloc[-1]
        return support, resistance

    def calculate_supertrend(self, high, low, close):
        # Placeholder implementation for SuperTrend
        if close.iloc[-1] > ta.SMA(close, timeperiod=10).iloc[-1]:
            return 1  # Bullish
        else:
            return -1  # Bearish

    def execute_trade(self, action):
        reward = 0
        log_disallowed_action = False  # Set to True to enable logging disallowed actions

        # Get positions for this symbol
        positions = mt5.positions_get(symbol=self.symbol)
        positions_count = len(positions)

        if positions_count > 0:
            # There is an open position
            position = positions[0]
            self.position_type = 'buy' if position.type == mt5.ORDER_TYPE_BUY else 'sell'

            # Calculate current profit
            current_profit = self.calculate_current_profit(position)

            # Stop-Loss at -$3
            if current_profit <= -3:
                result = self.close_position(position)
                if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                    # Log the trade closure
                    self.log_trade('Stop-Loss', position, current_profit)
                    reward += current_profit  # Negative reward
                    self.position = None
                    self.position_type = None

            # Take-Profit at $0.50
            elif current_profit >= 0.5:
                result = self.close_position(position)
                if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                    # Log the trade closure
                    self.log_trade('Take-Profit', position, current_profit)
                    reward += current_profit  # Positive reward
                    self.position = None
                    self.position_type = None

            elif action == 3:  # Close Position
                if current_profit > 0:
                    # Allow closing the position only if in profit
                    result = self.close_position(position)
                    if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                        # Log the trade closure
                        self.log_trade('Manual Close', position, current_profit)
                        reward += current_profit
                        self.position = None
                        self.position_type = None
                else:
                    # Disallow closing at a loss
                    if log_disallowed_action:
                        logging.warning("Attempted to close at a loss, action disallowed")
                    reward -= 1  # Penalize the agent
            else:
                # Hold position
                reward += 0  # No reward
        else:
            # No open position
            self.position_type = None
            if action == 0:  # Buy
                result = self.open_position('BUY')
                if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                    # Log the trade opening
                    self.log_trade('Open Buy', result, 0)
                    self.position = 'buy'
                    self.position_type = 'buy'
            elif action == 1:  # Sell
                result = self.open_position('SELL')
                if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                    # Log the trade opening
                    self.log_trade('Open Sell', result, 0)
                    self.position = 'sell'
                    self.position_type = 'sell'
            else:
                # Hold / No action
                self.position = None
                reward += 0  # No reward

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
            "comment": f"RL Agent",
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
            "comment": f"RL Agent Close",
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

    def log_trade(self, event_type, position, profit):
        # Calculate account balance growth
        current_balance = mt5.account_info().balance
        balance_change = current_balance - self.initial_balance

        # Log the trade event
        output = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Event: {event_type} | Symbol: {self.symbol} | " \
                 f"Position: {self.position_type} | Profit: {profit:.2f} | Account Balance: {current_balance:.2f} | " \
                 f"Balance Change: {balance_change:.2f}"
        print(output)

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def analyze_historical_data(self):
        # Combine data from all timeframes
        combined_data = pd.DataFrame()
        for tf in self.timeframes:
            data_files = glob.glob(f"analysis_data_{tf}_*.csv")
            for data_file in data_files:
                df = pd.read_csv(data_file)
                combined_data = pd.concat([combined_data, df], ignore_index=True)

        # Perform analysis to determine optimized levels
        # Placeholder for analysis logic
        # Example: Calculate average profitable trade size, loss size, etc.
        # You can implement advanced analysis or machine learning models here

        # For demonstration, set example optimized levels
        optimized_take_profit = 0.7  # Example value
        optimized_stop_loss = -2.5   # Example value

        # Output the optimized levels
        print(f"Optimized Take-Profit: ${optimized_take_profit:.2f}")
        print(f"Optimized Stop-Loss: ${optimized_stop_loss:.2f}")

# Main script
if __name__ == "__main__":
    # Initialize the environment
    env = TradingEnv(symbol='BTCUSD', timeframes=['M1', 'M5', 'M15'], lot_size=0.01, data_file_prefix='historical_data')
    model_prefix = "recurrent_ppo_model"
    training_start_time = time.time()
    training_duration = 24 * 60 * 60  # Train for 1 day (24 hours)
    save_interval = 60 * 60  # Save model every hour
    last_save_time = training_start_time

    # Model file with date
    current_date = datetime.now().strftime('%Y%m%d')
    model_path = f"{model_prefix}_{current_date}.zip"

    # Check if the model exists
    if os.path.exists(model_path):
        # Attempt to load the existing model
        try:
            model = RecurrentPPO.load(model_path, env=env)
            logging.info(f"Loaded existing model from {model_path}")
        except ValueError as e:
            logging.error(f"Failed to load existing model due to mismatched observation space: {e}")
            # Optionally, rename the old model file
            os.rename(model_path, model_path + "_old")
            # Create a new model
            model = RecurrentPPO('MlpLstmPolicy', env, verbose=0)
            logging.info("Created new RecurrentPPO model due to mismatched observation space")
    else:
        # Create a new model with LSTM policy
        model = RecurrentPPO('MlpLstmPolicy', env, verbose=0)
        logging.info("Created new RecurrentPPO model")

    # Live trading loop
    try:
        obs = env.reset()
        lstm_state = None  # Initialize LSTM state
        episode_start = True  # Track the start of episodes

        while True:
            current_time = time.time()
            elapsed_time = current_time - training_start_time

            if elapsed_time >= training_duration:
                logging.info("Training duration reached. Saving model and stopping training.")
                model.save(model_path)
                env.analyze_historical_data()
                break  # Exit the loop

            # Save model periodically
            if current_time - last_save_time >= save_interval:
                model.save(model_path)
                last_save_time = current_time
                logging.info("Model saved periodically.")

            # Get the previous LSTM state and episode start
            action, lstm_state = model.predict(obs, state=lstm_state, episode_start=episode_start, deterministic=False)
            episode_start = False  # After the first step, episode has started

            try:
                obs, reward, done, info = env.step(action)
            except Exception as e:
                logging.error(f"Error in environment: {e}")
                time.sleep(60)
                continue  # Skip to the next iteration

            # If done, reset the environment and LSTM state
            if done:
                obs = env.reset()
                lstm_state = None
                episode_start = True

            # Sleep for a short time to limit the rate of requests
            time.sleep(1)

    except KeyboardInterrupt:
        logging.info("Interrupted by user")

    finally:
        # Save model and historical data before exiting
        model.save(model_path)
        env.save_historical_data()
        env.close()
        mt5.shutdown()
        logging.info("Trading session ended")
