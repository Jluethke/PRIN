#!/usr/bin/env python3
"""
examples/torch_prin.py

Standalone PyTorch PRIN demo:
- trains PRIN_LSTM vs. BaselineLSTM on historical OHLCV data
- measures inference latency
- serializes run metadata
"""
import argparse
import sys

from neuroprin.train import run_pytorch_pipeline


def parse_args():
    p = argparse.ArgumentParser(
        description="Run the PyTorch PRIN vs. Baseline LSTM pipeline"
    )
    p.add_argument(
        "--symbols", nargs="+", required=True,
        help="Ticker symbols to fetch (e.g. AAPL MSFT GOOG)"
    )
    p.add_argument(
        "--start", type=str, required=True,
        help="Start date (YYYY-MM-DD)"
    )
    p.add_argument(
        "--end", type=str, required=True,
        help="End date (YYYY-MM-DD)"
    )
    p.add_argument(
        "--seq_len", type=int, default=20,
        help="Sequence length for LSTM inputs (default: 20)"
    )
    p.add_argument(
        "--test_size", type=float, default=0.2,
        help="Fraction of data to hold out for validation (default: 0.2)"
    )
    p.add_argument(
        "--batch_size", type=int, default=64,
        help="Batch size for training (default: 64)"
    )
    p.add_argument(
        "--epochs", type=int, default=50,
        help="Number of training epochs (default: 50)"
    )
    p.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "cuda"],
        help="Torch device to use (cpu or cuda, default: cpu)"
    )
    p.add_argument(
        "--output_dir", type=str, default="torch_runs",
        help="Directory to write models and run metadata (default: torch_runs)"
    )
    return p.parse_args()


def main():
    args = parse_args()

    # dispatch to the library function
    try:
        run_pytorch_pipeline(
            symbols=args.symbols,
            start_date=args.start,
            end_date=args.end,
            seq_len=args.seq_len,
            test_size=args.test_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            device=args.device,
            output_dir=args.output_dir,
        )
    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
