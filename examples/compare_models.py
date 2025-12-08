#!/usr/bin/env python3
"""
examples/compare_models.py

Compare multiple sequence-to-value models—including the full PRIN concept—
on the same raw data pipeline:
  – BaselineLSTM (standard nn.LSTM + dropout)
  – DPLSTM (dynamic-pruning LSTM)
  – PRIN_LSTM (Pruned Resonance Inference Network: pruning + attention + predictive coding)

Saves training histories, validation losses, inference latencies, and a summary JSON.
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from neuroprin.data import load_price_data, preprocess_data, compute_indicators
from neuroprin.models import BaselineLSTM, DPLSTM, PRIN_LSTM
from neuroprin.train import train_model
from neuroprin.utils import measure_inference_latency, save_run_metadata


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare BaselineLSTM, DPLSTM, and PRIN_LSTM on identical data"
    )
    p.add_argument("--symbols", nargs="+", required=True,
                   help="Ticker symbols, e.g. AAPL MSFT GOOG")
    p.add_argument("--start", type=str, required=True,
                   help="Start date YYYY-MM-DD")
    p.add_argument("--end", type=str, required=True,
                   help="End date YYYY-MM-DD")
    p.add_argument("--seq_len", type=int, default=20,
                   help="Sequence length (default: 20)")
    p.add_argument("--test_size", type=float, default=0.2,
                   help="Validation split fraction (default: 0.2)")
    p.add_argument("--batch_size", type=int, default=64,
                   help="Batch size (default: 64)")
    p.add_argument("--epochs", type=int, default=50,
                   help="Training epochs (default: 50)")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu","cuda"],
                   help="Torch device (cpu or cuda)")
    p.add_argument("--output_dir", type=str, default="comparison_runs",
                   help="Directory for outputs (default: comparison_runs)")
    return p.parse_args()


def prepare_data(symbols, start, end, seq_len, test_size):
    # 1) Load & preprocess & indicators
    df = load_price_data(symbols, start, end)
    df = preprocess_data(df)
    df = compute_indicators(df)
    closes = df["Close"].values

    # 2) Build sequences (N, seq_len, 1) and targets (N,)
    X, y = [], []
    for i in range(len(closes) - seq_len):
        X.append(closes[i : i + seq_len])
        y.append(closes[i + seq_len])
    X = np.expand_dims(np.array(X), -1)
    y = np.array(y)

    # 3) Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=False
    )

    # 4) DataLoaders
    train_ds = TensorDataset(torch.from_numpy(X_train).float(),
                             torch.from_numpy(y_train).float().unsqueeze(-1))
    val_ds   = TensorDataset(torch.from_numpy(X_val).float(),
                             torch.from_numpy(y_val).float().unsqueeze(-1))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    # 5) Sample input for latency tests
    sample_input = torch.from_numpy(X_val[:100]).float()
    return train_loader, val_loader, sample_input


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # Prepare data once
    train_loader, val_loader, sample_input = prepare_data(
        symbols=args.symbols,
        start=args.start,
        end=args.end,
        seq_len=args.seq_len,
        test_size=args.test_size,
    )

    results = {}

    # Model specs
    specs = {
        "BaselineLSTM": {
            "class": BaselineLSTM,
            "kwargs": {"input_size":1, "hidden_size":64, "output_size":1, "num_layers":1, "dropout_p":0.3}
        },
        "DPLSTM": {
            "class": DPLSTM,
            "kwargs": {"input_size":1, "hidden_size":64, "num_layers":1}
        },
        "PRIN_LSTM": {
            "class": PRIN_LSTM,
            "kwargs": {
                "input_size":1, "hidden_size":64, "output_size":1,
                "num_layers":1, "pruning_threshold":0.01,
                "dropout_p":0.5, "use_snn":False
            }
        }
    }

    for name, spec in specs.items():
        # Instantiate model
        model = spec["class"](**spec["kwargs"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        print(f"\n=== Training {name} ===")
        history = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            device,
            epochs=args.epochs
        )

        print(f"--- Measuring inference latency ({name}) ---")
        latency_ms = measure_inference_latency(
            model,
            sample_input,
            device=device,
            warmup=10,
            runs=50
        )

        results[name] = {
            "validation_history": history,
            "inference_latency_ms": latency_ms
        }

    # Save summary JSON
    summary = {
        "config": {
            "symbols": args.symbols,
            "start": args.start,
            "end": args.end,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "device": args.device
        },
        "results": results
    }
    summary_path = os.path.join(args.output_dir, "comparison_summary.json")
    save_run_metadata(summary_path, summary)
    print(f"\nAll done! Summary saved to {summary_path}")
