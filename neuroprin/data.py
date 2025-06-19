import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import mutual_info_regression
from hmmlearn.hmm import GaussianHMM

def load_price_data(filepath: str, index_col: str = None, parse_dates: bool = True) -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath, parse_dates=parse_dates, low_memory=False)
    except Exception as e:
        print(f"[ERROR] Failed to load CSV: {e}")
        raise

    if index_col is None:
        for col in df.columns:
            if 'date' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    df.set_index(col, inplace=True)
                    break
                except:
                    pass
    else:
        if index_col in df.columns:
            try:
                df[index_col] = pd.to_datetime(df[index_col], errors='coerce')
                df.set_index(index_col, inplace=True)
            except:
                pass

    df.dropna(axis=1, how='all', inplace=True)
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy().fillna(method='ffill').fillna(method='bfill')

def compute_indicators(df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
    df = df.copy()
    if target_col is None:
        target_col = df.columns[-1]

    if target_col in df:
        df[target_col] = pd.to_numeric(df[target_col].squeeze(), errors='coerce')
        df[f'{target_col}_SMA_10'] = df[target_col].rolling(10, min_periods=1).mean()
        df[f'{target_col}_SMA_50'] = df[target_col].rolling(50, min_periods=1).mean()
        df[f'{target_col}_EMA_10'] = df[target_col].ewm(span=10, adjust=False).mean()
        df[f'{target_col}_EMA_50'] = df[target_col].ewm(span=50, adjust=False).mean()
        df[f'{target_col}_ROC_10'] = df[target_col].pct_change(10)

        delta = df[target_col].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14, min_periods=1).mean()
        avg_loss = loss.rolling(14, min_periods=1).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        df[f'{target_col}_RSI_14'] = 100 - 100 / (1 + rs)
    return df

def compute_lyapunov_exponent(series: np.ndarray, window: int = 20, lag: int = 1) -> float:
    diffs = np.abs(series[lag:] - series[:-lag])
    lyap = np.mean(np.log(diffs + 1e-10)) / lag
    return max(lyap, 0.0)

def prune_data(df: pd.DataFrame, method: str = 'adaptive', base_alpha: float = 1.5, base_beta: float = 1.5, window: int = 14) -> pd.DataFrame:
    df = df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric columns available for pruning.")

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    chaos_scores = {}
    for col in numeric_cols:
        chaos_scores[col] = compute_lyapunov_exponent(df[col].dropna().values, window)

    best_col = max(chaos_scores, key=chaos_scores.get)
    chaos = chaos_scores[best_col]

    alpha = base_alpha * (1 + chaos)
    beta = base_beta * (1 + chaos)

    q_low, q_high = df[best_col].quantile([0.05, 0.95])
    extreme = (df[best_col] < q_low) | (df[best_col] > q_high)

    if method == 'static':
        mu = df[best_col].rolling(window, min_periods=1).mean()
        sigma = df[best_col].rolling(window, min_periods=1).std()
        thresh = alpha * sigma
        df['Pruned'] = np.where((np.abs(df[best_col] - mu) > thresh) | extreme, df[best_col], 0)
    else:
        if 'High' in df and 'Low' in df:
            ema = df[best_col].ewm(span=window, adjust=False).mean()
            atr = df['High'].rolling(window, min_periods=1).max() - df['Low'].rolling(window, min_periods=1).min()
            thresh = beta * atr
            df['Pruned'] = np.where((np.abs(df[best_col] - ema) > thresh) | extreme, df[best_col], 0)
        else:
            df['Pruned'] = df[best_col]

    return df.dropna()

def compute_fourier_resonance(df: pd.DataFrame, window_size: int = 20) -> pd.DataFrame:
    df = df.copy()
    features = [c for c in df.columns if c not in ['Pruned']]
    arr = df[features].values
    scores = np.zeros(len(arr))
    for i in range(window_size, len(arr)):
        w = arr[i - window_size:i]
        wn = (w - w.mean(axis=0)) / (w.std(axis=0) + 1e-9)
        fft_vals = np.abs(np.fft.fft(wn, axis=0))
        ps = fft_vals.mean(axis=1)
        scores[i] = ps[1:].max()
    df['Resonance'] = scores
    df['Resonance_Norm'] = (
        (df['Resonance'] - df['Resonance'].rolling(50, min_periods=1).mean()) /
        (df['Resonance'].rolling(50, min_periods=1).std() + 1e-9)
    )
    return df.dropna()

def compute_rqa_metrics(df: pd.DataFrame, window_size: int = 20) -> pd.DataFrame:
    df = df.copy()
    features = [c for c in df.columns if c not in ['Pruned', 'Resonance', 'Resonance_Norm']]
    arr = df[features].values
    rr = np.zeros(len(arr))
    for i in range(window_size, len(arr)):
        w = arr[i - window_size:i]
        wn = (w - w.mean(0)) / (w.std(0) + 1e-9)
        d = np.linalg.norm(wn[:, None] - wn[None, :], axis=2)
        rr[i] = (d < 0.1).mean()
    df['Recurrence_Rate'] = rr
    return df.dropna()

def prepare_sequences_with_prin_plus_plus(data_dict: dict, seq_length: int = 10, n_features: int = 5, n_regimes: int = 3):
    X_seqs, y_list, res_list, rec_list, feat_list = [], [], [], [], []

    for df in data_dict.values():
        df_p = prune_data(df)
        df_r = compute_fourier_resonance(df_p, window_size=seq_length)
        df_q = compute_rqa_metrics(df_r, window_size=seq_length)
        target_col = df_q.columns[-1]

        feats = df_q.select_dtypes(include=[np.number]).values
        target = df_q[target_col].shift(-1).values

        for i in range(len(df_q) - seq_length - 1):
            X_seqs.append(feats[i:i + seq_length])
            y_list.append(target[i + seq_length])
            res_list.append(df_q['Resonance_Norm'].iloc[i + seq_length])
            rec_list.append(df_q['Recurrence_Rate'].iloc[i + seq_length])
            feat_list.append(feats[i + seq_length])

    X = np.array(X_seqs)
    y = np.array(y_list)
    resonance = np.array(res_list)
    recurrence = np.array(rec_list)
    all_feats = np.array(feat_list)

    mi = mutual_info_regression(all_feats, y)
    idx = np.argsort(mi)[-n_features:]
    selected_feats = all_feats[:, idx]

    hmm = None
    regimes = None
    attempt = n_regimes
    while attempt >= 1:
        try:
            hmm = GaussianHMM(n_components=attempt, covariance_type='full', n_iter=100)
            hmm.fit(selected_feats)
            regimes = hmm.predict(selected_feats)
            break
        except ValueError as e:
            if "transmat_" in str(e) or "degenerate" in str(e).lower():
                print(f"[WARN] Degenerate HMM with {attempt} regimes. Retrying with fewer regimes...")
                attempt -= 1
            else:
                raise

    if regimes is None:
        raise RuntimeError("HMM training failed for all attempted regime counts.")

    return X, y, resonance, recurrence, regimes, attempt

class StockTradingEnv:
    def __init__(self, data: pd.DataFrame, window_size: int):
        self.data = data.reset_index(drop=True)
        self.window_size = window_size
        self.target_col = data.columns[-1] if 'Close' not in data else 'Close'
        self.reset()

    def reset(self):
        self.current_step = self.window_size
        self.position = 0
        self.total_reward = 0.0
        return self.data.iloc[self.current_step - self.window_size:self.current_step]

    def step(self, action: int):
        action = max(-1, min(1, action))
        prev_price = self.data[self.target_col].iloc[self.current_step - 1]
        curr_price = self.data[self.target_col].iloc[self.current_step]
        reward = self.position * (curr_price - prev_price)
        self.total_reward += reward
        self.position = action
        self.current_step += 1
        done = self.current_step >= len(self.data)
        state = None
        if not done:
            state = self.data.iloc[self.current_step - self.window_size:self.current_step]
        return state, reward, done, {}
