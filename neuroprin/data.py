import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

# ======================================
# CSV LOADING
# ======================================
def load_price_data(filepath: str, index_col: str = None, parse_dates: bool = True) -> pd.DataFrame:
    df = pd.read_csv(filepath, parse_dates=parse_dates, low_memory=False)

    if index_col is None:
        for col in df.columns:
            if "date" in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                    df.set_index(col, inplace=True)
                    break
                except:
                    pass
    else:
        if index_col in df.columns:
            df[index_col] = pd.to_datetime(df[index_col], errors="coerce")
            df.set_index(index_col, inplace=True)

    df.dropna(axis=1, how="all", inplace=True)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy().fillna(method="ffill").fillna(method="bfill")


# ======================================
# BASIC INDICATORS
# ======================================
def compute_indicators(df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
    df = df.copy()
    if target_col is None:
        target_col = df.columns[-1]

    if target_col in df:
        p = pd.to_numeric(df[target_col], errors="coerce")
        df[target_col] = p

        df[f"{target_col}_SMA_10"] = p.rolling(10, min_periods=1).mean()
        df[f"{target_col}_SMA_50"] = p.rolling(50, min_periods=1).mean()
        df[f"{target_col}_EMA_10"] = p.ewm(span=10, adjust=False).mean()
        df[f"{target_col}_EMA_50"] = p.ewm(span=50, adjust=False).mean()
        df[f"{target_col}_ROC_10"] = p.pct_change(10)

        delta = p.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        rs = gain.rolling(14, min_periods=1).mean() / (loss.rolling(14, min_periods=1).mean() + 1e-9)
        df[f"{target_col}_RSI_14"] = 100 - 100 / (1 + rs)

    return df


# ======================================
# CHAOS METRIC (Lyapunov Approx)
# ======================================
def compute_lyapunov_exponent(series: np.ndarray, window: int = 20, lag: int = 1) -> float:
    diffs = np.abs(series[lag:] - series[:-lag])
    lyap = np.mean(np.log(diffs + 1e-10)) / lag
    return max(lyap, 0.0)


# ======================================
# FOURIER RESONANCE
# ======================================
def compute_fourier_resonance(df: pd.DataFrame, window_size: int = 20) -> pd.DataFrame:
    df = df.copy()
    arr = df.values.astype(float)
    scores = np.zeros(len(arr))

    for i in range(window_size, len(arr)):
        w = arr[i - window_size:i]
        wn = (w - w.mean(0)) / (w.std(0) + 1e-9)

        fft_vals = np.abs(np.fft.fft(wn, axis=0))
        ps = fft_vals.mean(axis=1)

        scores[i] = ps[1:].max()

    df["Resonance"] = scores
    df["Resonance_Norm"] = (df["Resonance"] - df["Resonance"].rolling(50).mean()) / (
        df["Resonance"].rolling(50).std() + 1e-9
    )

    return df.dropna()


# ======================================
# FULL 10-CHANNEL RQA METRICS
# ======================================
def compute_full_rqa_metrics(df: pd.DataFrame, window_size: int = 20) -> pd.DataFrame:
    df = df.copy()
    arr = df.values.astype(float)

    # 10 RQA channels
    rqa_mat = np.zeros((len(arr), 10))

    for i in range(window_size, len(arr)):
        w = arr[i - window_size:i]
        wn = (w - w.mean(0)) / (w.std(0) + 1e-9)

        dist = np.linalg.norm(wn[:, None] - wn[None, :], axis=2)
        R = (dist < 0.1).astype(float)

        diagonal_lengths = []
        N = len(R)

        # Extract diagonal line lengths
        for k in range(-N + 1, N):
            diag = np.diag(R, k=k)
            if len(diag) > 1:
                count = 0
                for val in diag:
                    if val == 1:
                        count += 1
                    else:
                        if count > 1:
                            diagonal_lengths.append(count)
                        count = 0
                if count > 1:
                    diagonal_lengths.append(count)

        if len(diagonal_lengths) == 0:
            diagonal_lengths = [1]

        rr = R.mean()
        det = sum(l for l in diagonal_lengths if l > 2) / max(1, sum(diagonal_lengths))
        lmax = max(diagonal_lengths)
        lam = np.mean(np.sum(R == 1, axis=1))
        entr = -np.sum((np.array(diagonal_lengths) / sum(diagonal_lengths)) * np.log(
            (np.array(diagonal_lengths) / sum(diagonal_lengths)) + 1e-9
        ))
        tt = np.mean(diagonal_lengths)
        div = 1.0 / lmax
        ratio = det / (rr + 1e-9)
        var = np.var(diagonal_lengths)
        clust = np.mean(R.dot(R)) / (R.sum() + 1e-9)

        rqa_mat[i] = [
            rr, det, lam, entr, lmax, tt, div, ratio, var, clust
        ]

    # Construct dataframe
    cols = [
        "RQA_RR", "RQA_DET", "RQA_LAM", "RQA_ENTR", "RQA_LMAX",
        "RQA_TT", "RQA_DIV", "RQA_RATIO", "RQA_VAR", "RQA_CLUST"
    ]
    for j, c in enumerate(cols):
        df[c] = rqa_mat[:, j]

    return df.dropna()


# ======================================
# PRIN++ SEQUENCE GENERATOR (Aligned, Full-RQA)
# ======================================
def prepare_sequences_with_prin_plus_plus(
    df_dict,
    seq_length: int,
    n_features: int,
    n_regimes: int = 3
):

    print("\n==============================")
    print(">>> ENTERING PRIN++ BUILDER")
    print("==============================\n")

    df = df_dict["data"].copy()

    print(f"[DEBUG] Incoming df shape: {df.shape}")

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"[DEBUG] df after cleaning: {df.shape}")

    # ========== FEATURE TRUNCATION BLOCK ==========
    has_close = "Close" in df.columns
    print(f"[DEBUG] Close present: {has_close}")

    if df.shape[1] > n_features:
        print(f"[DEBUG] Truncating features from {df.shape[1]} -> {n_features}")

        if has_close:
            others = [c for c in df.columns if c != "Close"]
            k = max(0, min(len(others), n_features - 1))
            selected = others[:k] + ["Close"]
            df = df[selected]
        else:
            df = df.iloc[:, :n_features]

    print(f"[DEBUG] df after enforcing n_features: {df.shape}")

    # target index
    if "Close" in df:
        close_idx = df.columns.get_loc("Close")
    else:
        close_idx = df.shape[1] - 1

    print(f"[DEBUG] close_idx = {close_idx}")

    raw = df.values.astype(np.float32)
    T = len(raw)

    print(f"[DEBUG] raw array shape: {raw.shape}")

    # ========== CHAOS ==========
    chaos_source = raw[:, close_idx]

    chaos = np.array(
        [
            compute_lyapunov_exponent(chaos_source[:i], window=50) if i > 50 else 0.0
            for i in range(T)
        ],
        dtype=np.float32
    )
    print(f"[DEBUG] chaos shape: {chaos.shape}")

    # ========== RESONANCE & RQA ==========
    df_res = compute_fourier_resonance(df.copy()).reindex(df.index).ffill().bfill()
    df_rqa = compute_full_rqa_metrics(df.copy()).reindex(df.index).ffill().bfill()

    resonance = df_res["Resonance_Norm"].values.astype(np.float32)
    rqa_mat = df_rqa[[c for c in df_rqa.columns if c.startswith("RQA_")]].values.astype(np.float32)

    print(f"[DEBUG] resonance len: {len(resonance)}")
    print(f"[DEBUG] rqa_mat shape: {rqa_mat.shape}")

    # Normalize
    chaos = (chaos - chaos.mean()) / (chaos.std() + 1e-9)
    resonance = (resonance - resonance.mean()) / (resonance.std() + 1e-9)
    rqa_mat = (rqa_mat - rqa_mat.mean(0)) / (rqa_mat.std(0) + 1e-9)

    # ========== HMM ==========
    X_trunc = raw[:, :n_features]
    hmm = GaussianHMM(n_components=n_regimes, covariance_type="diag", n_iter=200, random_state=42)

    try:
        hmm.fit(X_trunc)
        regimes = hmm.predict(X_trunc).astype(np.int64)
    except Exception as e:
        print(f"[WARN] HMM failed: {e}")
        regimes = np.zeros(T, dtype=np.int64)

    regimes = np.clip(regimes, 0, n_regimes - 1)
    print(f"[DEBUG] regimes shape: {regimes.shape}")

    # ========== BUILD WINDOWS ==========
    X_seq = []
    y_seq = []
    chaos_seq = []
    res_seq = []
    regime_seq = []
    rqa_seq = []

    print(f"[DEBUG] Starting window generation. seq_length={seq_length}, T={T}")

    for t in range(seq_length, T):
        X_seq.append(raw[t - seq_length:t, :n_features])

        p_t = raw[t, close_idx]
        p_prev = raw[t - 1, close_idx]

        ret_t = (p_t - p_prev) / (p_prev + 1e-9)
        dir_t = 1.0 if ret_t > 0 else 0.0
        vol_t = abs(p_t - p_prev)

        y_seq.append([ret_t, dir_t, vol_t])
        chaos_seq.append(chaos[t - seq_length:t])
        res_seq.append(resonance[t - seq_length:t])
        regime_seq.append(regimes[t - seq_length:t])
        rqa_seq.append(rqa_mat[t - seq_length:t])

    print(f"[DEBUG] Final window counts: X={len(X_seq)}, y={len(y_seq)}, chaos={len(chaos_seq)}, res={len(res_seq)}, regimes={len(regime_seq)}, rqa={len(rqa_seq)}")

    # ====== EARLY EXIT BLOCK ======
    if len(X_seq) == 0:
        print("[ERROR] X_seq is empty, returning placeholder")
        return {
            "X": np.array([]),
            "y": np.array([]),
            "chaos": np.array([]),
            "resonance": np.array([]),
            "regimes": np.array([]),
            "rqa": np.array([]),
        }

    # ========== NORMALIZE X ==========
    X_seq = np.array(X_seq, dtype=np.float32)
    N, L, F = X_seq.shape

    print(f"[DEBUG] X_seq before norm: {X_seq.shape}")

    scaler = StandardScaler()
    X_seq = scaler.fit_transform(X_seq.reshape(N*L, F)).reshape(N, L, F)

    print(f"[DEBUG] X_seq after norm: {X_seq.shape}")

    # Convert
    X = X_seq  # <-- VALID FROM HERE ONWARD
    y_arr = np.array(y_seq, dtype=np.float32)
    chaos_arr = np.array(chaos_seq, dtype=np.float32)[..., None]
    res_arr = np.array(res_seq, dtype=np.float32)[..., None]
    regime_arr = np.array(regime_seq, dtype=np.int64)
    rqa_arr = np.array(rqa_seq, dtype=np.float32)

    print(f"[DEBUG] Final shapes BEFORE trimming:")
    print(f"    X: {X.shape}")
    print(f"    y: {y_arr.shape}")
    print(f"    chaos: {chaos_arr.shape}")
    print(f"    resonance: {res_arr.shape}")
    print(f"    regimes: {regime_arr.shape}")
    print(f"    rqa: {rqa_arr.shape}")

    # ========== ALIGN LENGTHS ==========
    N = min(len(X), len(y_arr), len(chaos_arr), len(res_arr), len(regime_arr), len(rqa_arr))
    print(f"[DEBUG] Final aligned N = {N}")

    return {
        "X": X[:N],
        "y": y_arr[:N],
        "chaos": chaos_arr[:N],
        "resonance": res_arr[:N],
        "regimes": regime_arr[:N],
        "rqa": rqa_arr[:N],
    }

class StockTradingEnv:
    """
    PRIN++ Enhanced RL Environment
    --------------------------------
    State consists of a concatenation of:
      - price window              → [L, n_features]
      - chaos window              → [L, 1]
      - resonance window          → [L, 1]
      - regime window             → [L, 1]
      - RQA window (10 channels)  → [L, 10]

    Total state shape = [L, n_features + 1 + 1 + 1 + 10]
    """

    def __init__(self, X, chaos, resonance, regimes, rqa, y, window_size):
        """
        X        : [N, L, n_features]
        chaos    : [N, L]
        resonance: [N, L]
        regimes  : [N, L]
        rqa      : [N, L, 10]
        y        : [N]
        """
        self.X = X
        self.chaos = chaos
        self.resonance = resonance
        self.regimes = regimes
        self.rqa = rqa
        self.y = y

        self.window_size = window_size
        self.idx = window_size
        self.position = 0
        self.total_reward = 0.0

    def _build_state(self, idx):
        """
        Build PRIN++ state tensor for the RL agent.
        """
        Xw = self.X[idx]                       # [L, F]
        cw = self.chaos[idx].reshape(-1, 1)    # [L, 1]
        rw = self.resonance[idx].reshape(-1, 1)# [L, 1]
        gw = self.regimes[idx].reshape(-1, 1)  # [L, 1]
        rqa = self.rqa[idx]                    # [L, 10]

        return np.concatenate([Xw, cw, rw, gw, rqa], axis=1)

    def reset(self):
        self.idx = self.window_size
        self.position = 0
        self.total_reward = 0.0
        return self._build_state(self.idx)

    def step(self, action):
        """
        action ∈ {-1, 0, 1}
        """
        action = int(max(-1, min(1, action)))

        prev_price = self.y[self.idx - 1]
        price = self.y[self.idx]

        reward = self.position * (price - prev_price)

        self.total_reward += reward
        self.position = action

        self.idx += 1
        done = self.idx >= len(self.y)

        next_state = None if done else self._build_state(self.idx)

        return next_state, reward, done, {}
