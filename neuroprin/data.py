"""
neuroprin/data.py

Core data-loading, preprocessing, environment, and technical-indicator utilities for NeuroPRIN.
"""
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import gym
from gym import spaces
import pandas_ta as ta


def load_price_data(
    symbols: List[str],
    start_date: str,
    end_date: str,
    interval: str = '1d',
    cache_dir: str = None
) -> pd.DataFrame:
    """
    Download and cache OHLCV data for given symbols.

    Returns a single DataFrame indexed by Datetime, with a MultiIndex columns (symbol, feature).
    """
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    all_data = []
    for symbol in symbols:
        cache_file = os.path.join(cache_dir, f"{symbol}_{interval}.parquet") if cache_dir else None
        if cache_file and os.path.exists(cache_file):
            df = pd.read_parquet(cache_file)
        else:
            df = yf.download(symbol, start=start_date, end=end_date, interval=interval, progress=False)
            if cache_file:
                df.to_parquet(cache_file)
        df['Symbol'] = symbol
        all_data.append(df.reset_index())
    df = pd.concat(all_data, ignore_index=True)
    df.set_index(['Date', 'Symbol'], inplace=True)
    df = df.sort_index()
    return df


def preprocess_data(
    df: pd.DataFrame,
    fill_method: str = 'ffill',
    dropna_threshold: float = 0.9
) -> pd.DataFrame:
    """
    Basic preprocessing:
    - Forward/backward fill missing values.
    - Drop columns with more than dropna_threshold fraction missing.
    - Drop remaining NaNs.
    """
    # Drop columns with too many NaNs
    thresh = int(len(df) * dropna_threshold)
    df = df.dropna(axis=1, thresh=thresh)
    df = df.fillna(method=fill_method).dropna()
    return df


def compute_indicators(
    df: pd.DataFrame,
    adx_length: int = 14,
    rsi_length: int = 14,
    bb_length: int = 20,
    bb_std: int = 2,
    atr_length: int = 14,
    volume_ma: int = 20
) -> pd.DataFrame:
    """
    Compute common technical indicators and append as new columns:
    - ADX, +DI, -DI
    - RSI
    - Bollinger Bands
    - ATR
    - Volume moving average
    """
    df = df.copy()
    # ADX and directional indicators
    adx = ta.adx(df['High'], df['Low'], df['Close'], length=adx_length)
    df['ADX'] = adx[f'ADX_{adx_length}']
    df['DI_PLUS'] = adx[f'DI_PLUS_{adx_length}']
    df['DI_MINUS'] = adx[f'DI_MINUS_{adx_length}']
    # RSI
    df['RSI'] = ta.rsi(df['Close'], length=rsi_length)

    # VWAP
    df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
    # Bollinger Bands
    bb = ta.bbands(df['Close'], length=bb_length, std=bb_std)
    df['BBL'] = bb[f'BBL_{bb_length}_{bb_std}.0']
    df['BBM'] = bb[f'BBM_{bb_length}_{bb_std}.0']
    df['BBU'] = bb[f'BBU_{bb_length}_{bb_std}.0']
    # ATR
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=atr_length)
    # Volume moving average
    df['VOL_MA'] = df['Volume'].rolling(volume_ma).mean()
    # Drop any rows with NA after indicator computation
    return df.dropna()


class StockTradingEnv(gym.Env):
    """
    A simple stock trading environment for OpenAI Gym.

    Observation:
        Type: Box(shape=(n_features,), dtype=np.float32)
    Actions:
        Type: Discrete(3)  # 0 = hold, 1 = buy, 2 = sell
    Reward:
        Profit and loss on each step.
    """
    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 10000.0,
        max_steps: int = None
    ):
        super().__init__()
        self.df = df.reset_index()
        self.symbols = self.df['Symbol'].unique().tolist()
        # features: OHLCV plus indicators
        self.feature_cols = [c for c in df.columns if c not in ['Symbol']]
        self.n_features = len(self.feature_cols)
        self.initial_balance = initial_balance
        self.max_steps = max_steps or len(self.df)

        # Define action and observation space
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_features + 2,), dtype=np.float32
        )

    def reset(self) -> np.ndarray:
        self.balance = self.initial_balance
        self.position = 0  # number of shares held
        self.current_step = 0
        self.trades = []
        return self._next_observation()

    def _next_observation(self) -> np.ndarray:
        row = self.df.iloc[self.current_step]
        obs = row[self.feature_cols].values.astype(np.float32)
        # Append current balance and position
        return np.concatenate([obs, [self.balance, self.position]]).astype(np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        assert self.action_space.contains(action)
        row = self.df.iloc[self.current_step]
        price = row['Close']
        reward = 0.0

        # Execute action
        if action == 1:  # buy
            if self.balance >= price:
                self.position += 1
                self.balance -= price
                self.trades.append((self.current_step, 'buy', price))
        elif action == 2:  # sell
            if self.position > 0:
                self.position -= 1
                self.balance += price
                self.trades.append((self.current_step, 'sell', price))

        # Calculate reward as change in net worth
        net_worth = self.balance + self.position * price
        if self.current_step > 0:
            prev_row = self.df.iloc[self.current_step - 1]
            prev_price = prev_row['Close']
            prev_net = self.balance + self.position * prev_price
            reward = net_worth - prev_net

        self.current_step += 1
        done = self.current_step >= self.max_steps
        obs = self._next_observation() if not done else None
        info = {'net_worth': net_worth}
        return obs, reward, done, info

    def render(self, mode='human'):
        net_worth = self.balance + self.position * self.df.iloc[self.current_step - 1]['Close']
        print(
            f"Step: {self.current_step}, Balance: {self.balance:.2f}, "
            f"Position: {self.position}, Net worth: {net_worth:.2f}"
        )

    def close(self):
        pass
