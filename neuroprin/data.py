import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import mutual_info_regression
from hmmlearn.hmm import GaussianHMM


def load_price_data(filepath: str, index_col: str = 'Date', parse_dates: bool = True) -> pd.DataFrame:
    """
    Load CSV price data, optionally parse dates and set index.
    """
    if parse_dates and index_col in pd.read_csv(filepath, nrows=0).columns:
        df = pd.read_csv(filepath, parse_dates=[index_col])
    else:
        df = pd.read_csv(filepath)
    if index_col in df.columns:
        df.set_index(index_col, inplace=True)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values via forward and backward fill.
    """
    df = df.copy()
    return df.fillna(method='ffill').fillna(method='bfill')




def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute standard technical indicators on OHLCV data.
    Adds SMA, EMA, ROC, Bollinger Bands, RSI, Stochastic %K, OBV.
    """
    df = df.copy()
    if 'Close' in df:
        df['SMA_10'] = df['Close'].rolling(10, min_periods=1).mean()
        df['SMA_50'] = df['Close'].rolling(50, min_periods=1).mean()
        df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['ROC_10'] = df['Close'].pct_change(10)
    if {'High','Low','Close'}.issubset(df.columns):
        df['BB_upper'] = df['SMA_10'] + 2*df['Close'].rolling(10, min_periods=1).std()
        df['BB_lower'] = df['SMA_10'] - 2*df['Close'].rolling(10, min_periods=1).std()
        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14, min_periods=1).mean()
        avg_loss = loss.rolling(14, min_periods=1).mean()
        rs = avg_gain/(avg_loss + 1e-9)
        df['RSI_14'] = 100 - 100/(1+rs)
        low_14 = df['Low'].rolling(14, min_periods=1).min()
        high_14 = df['High'].rolling(14, min_periods=1).max()
        df['Stoch_%K'] = 100*(df['Close']-low_14)/(high_14-low_14 + 1e-9)
    if 'Volume' in df:
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    return df


def compute_lyapunov_exponent(series: np.ndarray, window: int = 20, lag: int = 1) -> float:
    """
    Estimate a simple Lyapunov exponent for chaos measurement.
    """
    diffs = np.abs(series[lag:] - series[:-lag])
    lyap = np.mean(np.log(diffs + 1e-10)) / lag
    return max(lyap, 0.0)


def prune_data(df: pd.DataFrame, method: str = 'adaptive', base_alpha: float = 1.5, base_beta: float = 1.5, window: int = 14) -> pd.DataFrame:
    """
    Prune outliers based on static or adaptive thresholds.
    """
    df = df.copy()
    chaos = compute_lyapunov_exponent(df['Close'].values, window)
    alpha = base_alpha * (1 + chaos)
    beta = base_beta * (1 + chaos)
    q_low, q_high = df['Close'].quantile([0.05,0.95])
    extreme = (df['Close']<q_low)|(df['Close']>q_high)
    if method=='static':
        mu = df['Close'].rolling(window, min_periods=1).mean()
        sigma = df['Close'].rolling(window, min_periods=1).std()
        thresh = alpha * sigma
        df['Pruned'] = np.where((np.abs(df['Close']-mu)>thresh)|extreme, df['Close'], 0)
    else:
        ema = df['Close'].ewm(span=window, adjust=False).mean()
        atr = df['High'].rolling(window, min_periods=1).max() - df['Low'].rolling(window, min_periods=1).min()
        thresh = beta * atr
        df['Pruned'] = np.where((np.abs(df['Close']-ema)>thresh)|extreme, df['Close'], 0)
    return df.dropna()


def compute_fourier_resonance(df: pd.DataFrame, window_size: int = 20) -> pd.DataFrame:
    """
    Compute resonance scores over rolling windows.
    """
    df = df.copy()
    features = [c for c in df.columns if c not in ['Pruned']]
    arr = df[features].values
    scores = np.zeros(len(arr))
    for i in range(window_size, len(arr)):
        w = arr[i-window_size:i]
        wn = (w - w.mean(axis=0)) / (w.std(axis=0) + 1e-9)
        fft_vals = np.abs(np.fft.fft(wn, axis=0))
        ps = fft_vals.mean(axis=1)
        scores[i] = ps[1:].max()
    df['Resonance'] = scores
    df['Resonance_Norm'] = (
        (df['Resonance'] - df['Resonance'].rolling(50, min_periods=1).mean())
        / (df['Resonance'].rolling(50, min_periods=1).std() + 1e-9)
    )
    return df.dropna()


def compute_rqa_metrics(df: pd.DataFrame, window_size: int = 20) -> pd.DataFrame:
    """
    Compute recurrence rate metrics over rolling windows.
    """
    df = df.copy()
    features = [c for c in df.columns if c not in ['Pruned','Resonance','Resonance_Norm']]
    arr = df[features].values
    rr = np.zeros(len(arr))
    for i in range(window_size, len(arr)):
        w = arr[i-window_size:i]
        wn = (w - w.mean(0))/(w.std(0)+1e-9)
        d = np.linalg.norm(wn[:,None]-wn[None,:],axis=2)
        rr[i] = (d<0.1).mean()
    df['Recurrence_Rate'] = rr
    return df.dropna()


def prepare_sequences_with_prin_plus_plus(data_dict: dict, seq_length: int = 10,n_features: int = 5, n_regimes: int = 3):
    """
    Build sequences, compute MI-based feature selection, HMM regimes.
    Returns X, y, resonance, recurrence, regimes, n_regimes.
    """
    X_seqs, y_list, res_list, rec_list, feat_list = [], [], [], [], []
    for df in data_dict.values():
        df_p = prune_data(df)
        df_r = compute_fourier_resonance(df_p, window_size=seq_length)
        df_q = compute_rqa_metrics(df_r, window_size=seq_length)
        feats = df_q.select_dtypes(include=[np.number]).values
        target = df_q['Close'].shift(-1).values
        for i in range(len(df_q) - seq_length - 1):
            X_seqs.append(feats[i:i+seq_length])
            y_list.append(target[i+seq_length])
            res_list.append(df_q['Resonance_Norm'].iloc[i+seq_length])
            rec_list.append(df_q['Recurrence_Rate'].iloc[i+seq_length])
            feat_list.append(feats[i+seq_length])
    X = np.array(X_seqs)
    y = np.array(y_list)
    resonance = np.array(res_list)
    recurrence = np.array(rec_list)
    all_feats = np.array(feat_list)
    # Mutual information feature selection
    mi = mutual_info_regression(all_feats, y)
    idx = np.argsort(mi)[-n_features:]
    selected_feats = all_feats[:, idx]
    # HMM regime classification
    hmm = GaussianHMM(n_components=n_regimes, covariance_type='full', n_iter=100)
    hmm.fit(selected_feats)
    regimes = hmm.predict(selected_feats)
    return X, y, resonance, recurrence, regimes, n_regimes


class StockTradingEnv:
    """
    Simple trading environment to iterate over time-series data.
    State: window of previous observations.
    Action: buy (1), hold (0), sell (-1).
    Reward: change in price * position.
    """
    def __init__(self, data: pd.DataFrame, window_size: int):
        self.data = data.reset_index(drop=True)
        self.window_size = window_size
        self.reset()

    def reset(self):
        self.current_step = self.window_size
        self.position = 0  # 1 for long, -1 for short, 0 for flat
        self.total_reward = 0.0
        # initial state window
        return self.data.iloc[self.current_step-self.window_size:self.current_step]

    def step(self, action: int):
        # Clip action to {-1,0,1}
        action = max(-1, min(1, action))
        prev_price = self.data['Close'].iloc[self.current_step-1]
        curr_price = self.data['Close'].iloc[self.current_step]
        # compute reward: position * price change
        reward = self.position * (curr_price - prev_price)
        self.total_reward += reward
        # update position
        self.position = action
        # advance step
        self.current_step += 1
        done = self.current_step >= len(self.data)
        state = None
        if not done:
            state = self.data.iloc[self.current_step-self.window_size:self.current_step]
        return state, reward, done, {}

