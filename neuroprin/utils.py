"""
neuroprin/utils.py

Helper functions for NeuroPRIN:
- prepare_sequences: sliding-window sequence builder for LSTMs
- DirectionalMSELoss: asymmetric, variance-aware MSE loss
- prune_data: remove outlier price movements
- compute_acceleration_signal: compute acceleration (2nd diff)
- measure_inference_latency: benchmark model inference speed
- save_run_metadata: persist run metadata to JSON file
"""

import os
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

def prepare_sequences(
    X: np.ndarray,
    y: np.ndarray,
    sequence_length: int = 30
) -> tuple[np.ndarray, np.ndarray]:
    """
    Turn X (shape [N, feat_dim]) and y ([N,]) into
    sliding-window sequences for LSTM: ([N-L, L, feat_dim], [N-L]).
    """
    xs, ys = [], []
    for i in range(len(X) - sequence_length):
        xs.append(X[i : i + sequence_length])
        ys.append(y[i + sequence_length])
    return np.array(xs), np.array(ys)


class DirectionalMSELoss(nn.Module):
    """
    Enhanced Directional MSE Loss:
    Combines MSE, directional accuracy penalty,
    adaptive variance regularization, and asymmetric penalty.
    """
    def __init__(
        self,
        alpha: float = 0.01,
        epsilon: float = 1e-6,
        direction_weight: float = 0.1,
        asymmetry_factor: float = 2.0
    ):
        super().__init__()
        self.alpha = alpha
        self.epsilon = epsilon
        self.direction_weight = direction_weight
        self.asymmetry_factor = asymmetry_factor
        self.mse = nn.MSELoss()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mse_loss = self.mse(preds, targets)

        # Directional accuracy penalty
        pred_diff = preds[1:] - preds[:-1]
        targ_diff = targets[1:] - targets[:-1]
        dir_acc = (torch.sign(pred_diff) == torch.sign(targ_diff)).float().mean()
        dir_pen = (1 - dir_acc) * self.direction_weight

        # Adaptive variance regularization
        std = preds.std(unbiased=False)
        var_reg = -self.alpha * torch.log(std + self.epsilon)

        # Asymmetric penalty
        resid = preds - targets
        asym_pen = torch.where(
            resid > 0,
            resid**2,
            self.asymmetry_factor * (resid**2)
        ).mean()

        return mse_loss + dir_pen + var_reg + asym_pen


def prune_data(
    df: pd.DataFrame,
    zscore_threshold: float = 3.0,
    adaptive: bool = False
) -> pd.DataFrame:
    """
    Remove extreme price movements.
    If adaptive=True, threshold adjusts for volatility.
    """
    def _prune(group: pd.DataFrame) -> pd.DataFrame:
        returns = group['Close'].pct_change().dropna()
        mean, std = returns.mean(), returns.std()
        thresh = zscore_threshold * (1 + std) if adaptive else zscore_threshold
        zs = (returns - mean) / (std + 1e-8)
        return group.loc[zs.abs() < thresh]

    if isinstance(df.index, pd.MultiIndex) and 'Symbol' in df.index.names:
        return df.groupby(level='Symbol', group_keys=False).apply(_prune)
    pruned = _prune(df.reset_index())
    return pruned.set_index(df.index.names)


def compute_acceleration_signal(
    df: pd.DataFrame,
    feature: str = 'Close'
) -> pd.DataFrame:
    """
    Compute acceleration (2nd derivative) of a price series.
    Adds column 'Acceleration'.
    """
    df = df.copy()
    df['Diff1'] = df[feature].diff()
    df['Acceleration'] = df['Diff1'].diff()
    return df.drop(columns=['Diff1']).dropna()


def measure_inference_latency(
    model,
    sample_input,
    device: torch.device = torch.device('cpu'),
    warmup: int = 10,
    runs: int = 100,
    backend: str = 'torch'
) -> float:
    """
    Benchmark inference time (ms) for PyTorch or Keras models.
    """
    if backend == 'torch':
        model.to(device).eval()
        inp = sample_input.to(device)
        with torch.no_grad():
            for _ in range(warmup): _ = model(inp)
            times = []
            for _ in range(runs):
                start = time.perf_counter()
                _ = model(inp)
                if device.type == 'cuda': torch.cuda.synchronize()
                end = time.perf_counter()
                times.append((end - start) * 1000)
        return float(np.mean(times))
    elif backend == 'keras':
        import tensorflow as tf
        for _ in range(warmup): _ = model(sample_input, training=False)
        times = []
        for _ in range(runs):
            start = time.perf_counter()
            _ = model(sample_input, training=False)
            tf.experimental.sync_devices()
            end = time.perf_counter()
            times.append((end - start) * 1000)
        return float(np.mean(times))
    else:
        raise ValueError(f"Unknown backend: {backend}")


def save_run_metadata(
    filepath: str,
    metadata: dict,
    timestamp: bool = True
) -> None:
    """
    Save metadata to JSON. Adds timestamp if requested.
    """
    if timestamp:
        metadata["_saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2)
