import MetaTrader5 as mt5
import gym
import numpy as np
import talib as ta  # Requires TA-Lib for technical indicators
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

class TradingEnv(gym.Env):
    def __init__(self, timeframe='M1'):
        super(TradingEnv, self).__init__()

        # Initialize MetaTrader5
        if not mt5.initialize():
            logging.error("MetaTrader5 initialization failed")
            quit()

        # Login to MetaTrader5 with credentials
        login = 62061021  # Your account number
        password = 'Mailsam96@'  # Your password
        server = 'PepperstoneUK-Demo'  # Your server name

        if not mt5.login(login, password=password, server=server):
            logging.error(f"Failed to login to MetaTrader5, error: {mt5.last_error()}")
            quit()

        # Define action and observation space
        self.action_space = gym.spaces.Discrete(3)  # Buy, sell, or hold
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)

        # Set timeframe based on user input
        self.timeframe = getattr(mt5, f'TIMEFRAME_{timeframe}')

        # Initialize price tracking
        self.current_price = None
        self.entry_price = None
        self.trade_open = False  # Track if a trade is open
        self.position = None  # "buy" or "sell"

    def _calculate_reward(self, action):
        if self.entry_price is None:
            logging.error("Error: Entry price not initialized correctly")
            return 0

        # Reward based on Buy/Sell trades
        if self.position == "buy":
            reward = self.current_price - self.entry_price  # Profit for Buy: Exit - Entry
        elif self.position == "sell":
            reward = self.entry_price - self.current_price  # Profit for Sell: Entry - Exit
        else:
            reward = 0

        logging.info(f"Action: {action}, Position: {self.position}, Reward: {reward}")
        return reward

    def _check_exit_conditions(self, close_prices, ema50, supertrend, adx):
        # Define exit conditions based on market indicators
        # SuperTrend flip
        if self.position == "buy" and supertrend == -1:
            logging.info("Exit condition met: SuperTrend flipped bearish.")
            return True
        if self.position == "sell" and supertrend == 1:
            logging.info("Exit condition met: SuperTrend flipped bullish.")
            return True

        # EMA crossover
        if self.position == "buy" and close_prices[-1] < ema50:
            logging.info("Exit condition met: Price crossed below EMA.")
            return True
        if self.position == "sell" and close_prices[-1] > ema50:
            logging.info("Exit condition met: Price crossed above EMA.")
            return True

        # ADX shows weakening trend
        if adx < 20:  # Example: ADX below 20 signals weak trend
            logging.info("Exit condition met: ADX is below threshold.")
            return True

        return False

    def step(self, action):
        # Retrieve market data (XAUUSD)
        xauusd_rates = mt5.copy_rates_from_pos("XAUUSD", self.timeframe, 0, 500)

        if xauusd_rates is None or len(xauusd_rates) == 0:
            logging.error("Error: No market data retrieved for XAUUSD")
            return np.array([0] * 9), 0, True, {}

        # Extract the relevant fields as floats
        close_prices = np.array([rate['close'] for rate in xauusd_rates], dtype=np.float64)

        # Calculate technical indicators
        ema50 = ta.EMA(close_prices, timeperiod=50)[-1]  # 50 EMA
        rsi = ta.RSI(close_prices, timeperiod=14)[-1]  # RSI
        adx = ta.ADX(np.array([rate['high'] for rate in xauusd_rates], dtype=np.float64),
                     np.array([rate['low'] for rate in xauusd_rates], dtype=np.float64),
                     close_prices, timeperiod=14)[-1]  # ADX

        # Simulate SuperTrend (basic example)
        supertrend = np.where(close_prices[-1] > ema50, 1, -1)  # 1 for bullish, -1 for bearish

        self.current_price = close_prices[-1]

        # Create observation array with 9 values
        obs = np.array([close_prices[-1], ema50, rsi, adx, supertrend, np.min(close_prices),
                        np.max(close_prices), np.mean(close_prices), np.std(close_prices)])

        if self.trade_open:
            # Check exit conditions dynamically
            if self._check_exit_conditions(close_prices, ema50, supertrend, adx):
                reward = self._calculate_reward(action)  # Exit trade and calculate reward
                self.trade_open = False  # Close the trade
                logging.info(f"Trade closed. Reward: {reward} for {self.timeframe}")
            else:
                reward = 0  # Keep trade open, no reward yet
        else:
            reward = 0  # No trade open, no reward

        # Action: 0 = Buy, 1 = Sell, 2 = Hold
        if not self.trade_open:
            if action == 0:  # Buy
                self.entry_price = self.current_price
                self.position = "buy"
                self.trade_open = True
                logging.info(f"Opening Buy trade at {self.current_price} for {self.timeframe}")
            elif action == 1:  # Sell
                self.entry_price = self.current_price
                self.position = "sell"
                self.trade_open = True
                logging.info(f"Opening Sell trade at {self.current_price} for {self.timeframe}")

        done = False  # Or any other logic you need to finish the episode
        return obs, reward, done, {}

    def reset(self):
        self.current_price = None
        self.entry_price = None
        self.trade_open = False
        self.position = None
        logging.info("Environment reset")
        return np.zeros(9, dtype=np.float32)

    def close(self):
        mt5.shutdown()
