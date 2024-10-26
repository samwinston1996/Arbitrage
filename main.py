import MetaTrader5 as mt5
import gym
from gym import spaces
import numpy as np
import talib as ta
from stable_baselines3 import PPO
import logging
import time
import pandas as pd
from datetime import datetime

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Initialize MetaTrader5
if not mt5.initialize():
    logging.error("MetaTrader5 initialization failed")
    quit()

# Login to MetaTrader5 with credentials
login = 62061021  # Replace with your account number
password = 'Mailsam96@'  # Replace with your password
server = 'PepperstoneUK-Demo'  # Replace with your server name

if not mt5.login(login, password=password, server=server):
    logging.error(f"Failed to login to MetaTrader5, error: {mt5.last_error()}")
    mt5.shutdown()
    quit()
else:
    logging.info("MetaTrader5 initialized and logged in")

class TradingEnv(gym.Env):
    def __init__(self, symbol='XAUUSD', timeframe='M1', lot_size=0.01):
        super(TradingEnv, self).__init__()

        # Set symbol and timeframe
        self.symbol = symbol
        self.timeframe = getattr(mt5, f'TIMEFRAME_{timeframe}')
        self.timeframe_name = timeframe
        self.lot_size = lot_size

        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # Buy, Sell, or Hold
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)

        # For storing previous indicators to compute changes
        self.prev_observation = None

        # Initialize account info
        self.account_info = mt5.account_info()
        if self.account_info is None:
            logging.error("Failed to get account info")
            mt5.shutdown()
            quit()

    def reset(self):
        self.prev_observation = None
        logging.info(f"Environment reset for timeframe {self.timeframe_name}")
        return np.zeros(11, dtype=np.float32)

    def step(self, action):
        # Retrieve market data
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, 500)

        if rates is None or len(rates) == 0:
            logging.error("No market data retrieved")
            return np.zeros(11), 0, True, {}

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

        # Execute action
        reward = self.execute_trade(action, current_price)

        # Log current market price and action
        self.log_market_data(obs, action, reward)

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

        if positions_count < 1:
            # Can open a new position
            if action == 0:  # Buy
                result = self.open_position('BUY')
                if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                    logging.info(f"Opened Buy position at {result.price} on {self.symbol} ({self.timeframe_name})")
                    self.position = 'buy'
            elif action == 1:  # Sell
                result = self.open_position('SELL')
                if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                    logging.info(f"Opened Sell position at {result.price} on {self.symbol} ({self.timeframe_name})")
                    self.position = 'sell'
            else:
                # Hold
                pass
        else:
            # Manage existing positions
            for position in positions:
                # Determine the position's current profit
                if position.type == mt5.ORDER_TYPE_BUY:
                    current_price = mt5.symbol_info_tick(self.symbol).bid
                    current_profit = (current_price - position.price_open) * position.volume * mt5.symbol_info(self.symbol).trade_contract_size
                else:  # SELL position
                    current_price = mt5.symbol_info_tick(self.symbol).ask
                    current_profit = (position.price_open - current_price) * position.volume * mt5.symbol_info(self.symbol).trade_contract_size

                # Check exit conditions
                if self.check_exit_conditions(current_profit, position):
                    result = self.close_position(position)
                    if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                        logging.info(f"Closed {'Buy' if position.type == mt5.ORDER_TYPE_BUY else 'Sell'} position at {current_price} with profit {position.profit} ({self.timeframe_name})")
                        reward += position.profit  # Sum up profits from closed positions

        return reward

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
        if symbol_info.filling_mode & mt5.SYMBOL_FILLING_FOK:
            filling_modes.append(mt5.ORDER_FILLING_FOK)
        if symbol_info.filling_mode & mt5.SYMBOL_FILLING_IOC:
            filling_modes.append(mt5.ORDER_FILLING_IOC)
        if symbol_info.filling_mode & mt5.SYMBOL_FILLING_RETURN:
            filling_modes.append(mt5.ORDER_FILLING_RETURN)
        if filling_modes:
            return filling_modes[0]  # Use the first supported filling mode
        else:
            logging.error(f"No acceptable filling mode found for symbol {self.symbol}")
            return None

    def check_exit_conditions(self, current_profit, position):
        profit_target = 1.0  # At least $1 profit
        stop_loss = -10.0    # Maximum loss per trade is $10

        # Calculate trade duration
        position_duration = time.time() - position.time
        minimum_duration = 60  # 60 seconds; adjust as needed

        if (current_profit >= profit_target or current_profit <= stop_loss) and position_duration >= minimum_duration:
            return True
        else:
            return False

    def log_market_data(self, obs, action, reward):
        data = {
            'Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Symbol': self.symbol,
            'Timeframe': self.timeframe_name,
            'Price': obs[0],
            'EMA50': obs[1],
            'RSI': obs[2],
            'ADX': obs[3],
            'SuperTrend': obs[4],
            'Action': ['Buy', 'Sell', 'Hold'][action],
            'Position': self.position,
            'Reward': reward
        }
        df = pd.DataFrame([data])
        print(df.to_string(index=False))

    def render(self, mode='human'):
        pass

    def close(self):
        pass

# Initialize environments and agents for multiple timeframes
timeframes = ['M1', 'M5', 'M15']
envs = {}
models = {}
max_trades_per_timeframe = 3

for tf in timeframes:
    envs[tf] = TradingEnv(symbol='XAUUSD', timeframe=tf)
    models[tf] = PPO('MlpPolicy', envs[tf], verbose=0)

# Live trading loop
try:
    obs = {tf: envs[tf].reset() for tf in timeframes}
    while True:
        for tf in timeframes:
            action, _states = models[tf].predict(obs[tf], deterministic=True)
            obs[tf], reward, done, info = envs[tf].step(action)
            if reward != 0:
                # Train the model with the obtained reward
                models[tf].learn(total_timesteps=1)
            # Sleep for 5 seconds to limit the rate of requests
            time.sleep(5)

except KeyboardInterrupt:
    logging.info("Interrupted by user")

except Exception as e:
    logging.error(f"An error occurred: {e}")

finally:
    for tf in timeframes:
        envs[tf].close()
    mt5.shutdown()
    logging.info("Trading session ended")
