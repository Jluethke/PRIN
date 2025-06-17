"""
NeuroPRIN: Pruned Resonance Inference Network core package initializer.

This package provides:
- Neural network model definitions for PRIN.
- Data loading, preprocessing, and environment utilities.
- Training loops with support for pruning, EWC, DP, and hyperparameter search.
- Utility functions for pruning heuristics, signal computation, and performance measurement.

"""

__version__ = "0.1.0"

# Models
from .models import (
    DPLSTM,
    PRIN_LSTM,
    BaselineLSTM,
    TemporalConvBlock,
    FourierTransformLayer,
)

# Data loading and preprocessing
from .data import (
    load_price_data,
    preprocess_data,
    StockTradingEnv,
    compute_indicators,
)

# Training and experimentation
from .train import (
    train_model,
    grid_search,
    run_keras_models,
    run_pytorch_pipeline,
)

# Utilities
from .utils import (
    prune_data,
    compute_acceleration_signal,
    measure_inference_latency,
    save_run_metadata,
    DirectionalMSELoss,
)

__all__ = [
    # Version
    "__version__",
    # Models
    "DPLSTM", "PRIN_LSTM", "BaselineLSTM",
    "TemporalConvBlock", "FourierTransformLayer", "DirectionalMSELoss",
    # Data
    "load_price_data", "preprocess_data", "StockTradingEnv", "compute_indicators",
    # Training
    "train_model", "grid_search", "run_keras_models", "run_pytorch_pipeline",
    # Utilities
    "prune_data", "compute_acceleration_signal", "measure_inference_latency", "save_run_metadata",
]
