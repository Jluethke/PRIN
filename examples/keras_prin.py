#!/usr/bin/env python3
"""
examples/keras_prin.py

Standalone demo for the TensorFlow/Keras PRIN LSTM pipeline.
Loads price data, computes indicators, trains an LSTM model, and saves outputs.
"""
import argparse
import os
from neuroprin.train import run_keras_models

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Keras PRIN LSTM demo."
    )
    parser.add_argument(
        "--symbols", 
        nargs="+", 
        required=True,
        help="Ticker symbols to fetch (e.g. AAPL MSFT GOOG)"
    )
    parser.add_argument(
        "--start", 
        type=str, 
        required=True,
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end", 
        type=str, 
        required=True,
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--seq_len", 
        type=int, 
        default=20,
        help="Sequence length for LSTM input (default: 20)"
    )
    parser.add_argument(
        "--test_size", 
        type=float, 
        default=0.2,
        help="Fraction of data to hold out for validation (default: 0.2)"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=50,
        help="Number of training epochs (default: 50)"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=32,
        help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="keras_runs",
        help="Directory to save model and history (default: keras_runs)"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    run_keras_models(
        symbols=args.symbols,
        start_date=args.start,
        end_date=args.end,
        seq_len=args.seq_len,
        test_size=args.test_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )

if __name__ == "__main__":
    main()
