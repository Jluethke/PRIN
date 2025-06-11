"""
neuroprin/utils.py

Helper functions for NeuroPRIN:
- prune_data: remove outlier price movements
- compute_acceleration_signal: compute acceleration (2nd diff) of prices
- measure_inference_latency: benchmark model inference speed
- save_run_metadata: persist run metadata to JSON file
"""

import os
import json
import time
import numpy as np
import pandas as pd
import torch


def prune_data(
    df: pd.DataFrame,
    zscore_threshold: float = 3.0
) -> pd.DataFrame:
    """
    Remove extreme price movements per symbol based on z-score of pct change.
    """
    def _prune(group: pd.DataFrame) -> pd.DataFrame:
        returns = group['Close'].pct_change().dropna()
        mean = returns.mean()
        std = returns.std()
        zscores = (returns - mean) / (std + 1e-8)
        valid = zscores.abs() < zscore_threshold
        return group.loc[valid.index.intersection(group.index)]

    if isinstance(df.index, pd.MultiIndex) and 'Symbol' in df.index.names:
        return df.groupby(level='Symbol', group_keys=False).apply(_prune)
    else:
        pruned = _prune(df.reset_index())
        return pruned.set_index(df.index.names)


def compute_acceleration_signal(
    df: pd.DataFrame,
    feature: str = 'Close'
) -> pd.DataFrame:
    """
    Compute acceleration (2nd derivative) of a price series.
    Adds column 'Acceleration' to DataFrame.
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
    Unified inference latency measurement for both PyTorch and Keras models.
    """
    if backend == 'torch':
        model.to(device).eval()
        inp = sample_input.to(device)

        with torch.no_grad():
            for _ in range(warmup):
                _ = model(inp)

            times = []
            for _ in range(runs):
                start = time.perf_counter()
                _ = model(inp)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                end = time.perf_counter()
                times.append((end - start) * 1000)
        return float(np.mean(times))

    elif backend == 'keras':
        import tensorflow as tf
        for _ in range(warmup):
            _ = model(sample_input, training=False)

        times = []
        for _ in range(runs):
            start = time.perf_counter()
            _ = model(sample_input, training=False)
            tf.experimental.sync_devices()  # Optional sync for TF > 2.9
            end = time.perf_counter()
            times.append((end - start) * 1000)
        return float(np.mean(times))

    else:
        raise ValueError(f"Unknown backend: {backend}")

def prune_data(
    df: pd.DataFrame,
    zscore_threshold: float = 3.0,
    adaptive: bool = False
) -> pd.DataFrame:
    """
    Remove extreme price movements.
    If adaptive=True, adjusts threshold dynamically based on series volatility.
    """
    def _prune(group: pd.DataFrame) -> pd.DataFrame:
        returns = group['Close'].pct_change().dropna()
        mean = returns.mean()
        std = returns.std()
        adaptive_z = zscore_threshold * (1 + std) if adaptive else zscore_threshold
        zscores = (returns - mean) / (std + 1e-8)
        valid = zscores.abs() < adaptive_z
        return group.loc[valid.index.intersection(group.index)]

    if isinstance(df.index, pd.MultiIndex) and 'Symbol' in df.index.names:
        return df.groupby(level='Symbol', group_keys=False).apply(_prune)
    else:
        pruned = _prune(df.reset_index())
        return pruned.set_index(df.index.names)


def save_run_metadata(
    filepath: str,
    metadata: dict,
    timestamp: bool = True
) -> None:
    """
    Save metadata dictionary to a JSON file, optionally adding timestamp.
    """
    if timestamp:
        metadata["_saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2)

