"""
neuroprin/train.py

Training loops, hyperparameter search, and framework entry points for NeuroPRIN.
Includes:
- train_model (PyTorch) with optional EWC and DP noise
- grid_search (PyTorch) via ParameterGrid
- run_keras_models: TensorFlow/Keras PRIN demo
- run_pytorch_pipeline: PyTorch PRIN training & evaluation
"""
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import ParameterGrid, train_test_split
import optuna

from .models import PRIN_LSTM, BaselineLSTM
from .data import load_price_data, preprocess_data, compute_indicators
from .utils import prune_data, measure_inference_latency, save_run_metadata

# TensorFlow for Keras pipeline
import tensorflow as tf


from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 50,
    use_ewc: bool = False,
    ewc_lambda: float = 0.4,
    dp_noise_std: float = 0.0
) -> dict:
    """
    Train a PyTorch model with optional EWC and differential privacy noise.
    Returns training history and final model.
    """
    model.to(device)
    history = {'train_loss': [], 'val_loss': []}
    # Save original parameters for EWC
    if use_ewc:
        orig_params = {n: p.clone().detach() for n, p in model.named_parameters()}
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, y)
            # EWC penalty
            if use_ewc:
                ewc_penalty = 0.0
                for n, p in model.named_parameters():
                    ewc_penalty += ((p - orig_params[n])**2).sum()
                loss = loss + ewc_lambda * ewc_penalty
            loss.backward()
            # DP noise
            if dp_noise_std > 0.0:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad += torch.randn_like(p.grad) * dp_noise_std
            optimizer.step()
            train_losses.append(loss.item())
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                preds = model(X)
                val_losses.append(criterion(preds, y).item())
        history['train_loss'].append(np.mean(train_losses))
        history['val_loss'].append(np.mean(val_losses))
    return history


def grid_search(
    model_class,
    param_grid: dict,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    epochs: int = 10,
    batch_size: int = 64
) -> tuple:
    """
    Simple grid search over hyperparameters for PyTorch models.
    Returns best_params, best_score.
    """
    best_score = float('inf')
    best_params = None
    for params in ParameterGrid(param_grid):
        # Prepare data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        )
        val_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32)
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)
        # Init model, criterion, optimizer
        model = model_class(
            **{k: params[k] for k in ['input_size','hidden_size','output_size','num_layers'] if k in params}
        )
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params.get('lr', 1e-3))
        history = train_model(
            model, train_loader, val_loader,
            criterion, optimizer, device,
            epochs=epochs
        )
        score = min(history['val_loss'])
        if score < best_score:
            best_score = score
            best_params = params
    return best_params, best_score


def run_keras_models(
    symbols: list,
    start_date: str,
    end_date: str,
    seq_len: int = 20,
    test_size: float = 0.2,
    epochs: int = 50,
    batch_size: int = 32,
    output_dir: str = 'keras_runs'
) -> None:
    """
    NeuroPRIN v0.2.0: Keras PRIN demo pipeline.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1️⃣ Load, preprocess, compute indicators
    df = load_price_data(symbols, start_date, end_date)
    df = preprocess_data(df)
    df = compute_indicators(df)

    # 2️⃣ Sequence preparation
    data = df['Close'].values.reshape(-1, 1)
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=False
    )

    # 3️⃣ Build Keras model
    model = Sequential([
        LSTM(64, input_shape=(seq_len, 1)),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse')

    # 4️⃣ Add strong checkpointing
    checkpoint_path = os.path.join(output_dir, "keras_prin_model.keras")
    checkpoint_cb = ModelCheckpoint(checkpoint_path, save_best_only=True)
    early_stop_cb = EarlyStopping(patience=5, restore_best_weights=True)

    # 5️⃣ Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop_cb, checkpoint_cb],
        verbose=2
    )

    # 6️⃣ Export history to JSON
    history_path = os.path.join(output_dir, 'history.json')
    with open(history_path, 'w') as f:
        json.dump(history.history, f)

    print(f"✅ Keras model saved: {checkpoint_path}")
    print(f"✅ Training history saved: {history_path}")

def run_pytorch_pipeline(
    symbols: list,
    start_date: str,
    end_date: str,
    seq_len: int = 20,
    test_size: float = 0.2,
    batch_size: int = 64,
    epochs: int = 50,
    device: str = 'cpu',
    output_dir: str = 'torch_runs'
) -> None:
    """
    PyTorch PRIN: train PRIN_LSTM vs BaselineLSTM, measure latency, save metadata.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device)
    # Data
    df = load_price_data(symbols, start_date, end_date)
    df = preprocess_data(df)
    df = compute_indicators(df)
    prices = df['Close'].values
    # Prepare sequences
    X, y = [], []
    for i in range(len(prices) - seq_len):
        X.append(prices[i:i+seq_len])
        y.append(prices[i+seq_len])
    X = np.expand_dims(np.array(X), -1)
    y = np.array(y)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    # DataLoaders
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    # Models
    prin = PRIN_LSTM(input_size=1, hidden_size=64, output_size=1, num_layers=1)
    base = BaselineLSTM(input_size=1, hidden_size=64, output_size=1)
    # Train
    criterion = nn.MSELoss()
    optim_prin = torch.optim.Adam(prin.parameters(), lr=1e-3)
    optim_base = torch.optim.Adam(base.parameters(), lr=1e-3)
    hist_prin = train_model(prin, train_loader, val_loader, criterion, optim_prin, device, epochs=epochs)
    hist_base = train_model(base, train_loader, val_loader, criterion, optim_base, device, epochs=epochs)
    # Latency
    lat_prin = measure_inference_latency(prin, torch.tensor(X_val[:100], dtype=torch.float32), device)
    lat_base = measure_inference_latency(base, torch.tensor(X_val[:100], dtype=torch.float32), device)
    # Save metadata
    meta = {
        'prin_history': hist_prin,
        'base_history': hist_base,
        'prin_latency': lat_prin,
        'base_latency': lat_base,
        'config': {
            'symbols': symbols,
            'seq_len': seq_len,
            'batch_size': batch_size,
            'epochs': epochs,
        }
    }
    save_run_metadata(os.path.join(output_dir, 'run_meta.json'), meta)
