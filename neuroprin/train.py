"""
neuroprin/train.py

Training loops, hyperparameter search, and framework entry points for NeuroPRIN.
Includes:
- train_model (PyTorch) with optional EWC and DP noise
- grid_search (PyTorch) via ParameterGrid
- run_keras_models: TensorFlow/Keras PRIN demo
- run_pytorch_pipeline: PyTorch PRIN training & evaluation

UPDATED: Integrates PRIN++ full sequence builder so NeuroPRINv4 trains with
chaos, resonance, AND regime sequences as required by the PRIN theorems.
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

from .models import PRIN_LSTM, BaselineLSTM, NeuroPRINv4
from .data import (
    load_price_data,
    preprocess_data,
    compute_indicators,
    compute_lyapunov_exponent,
    compute_fourier_resonance,
)
from .utils import prune_data, measure_inference_latency, save_run_metadata

### >>> ADDED: import PRIN++ sequence builder
from .data import prepare_sequences_with_prin_plus_plus

import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam


class MultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()  # for direction classification

    def forward(self, pred, target):
        # pred: [B,3], target: [B,3]
        ret_pred, dir_pred, vol_pred = pred[:,0], pred[:,1], pred[:,2]
        ret_t,   dir_t,   vol_t     = target[:,0], target[:,1], target[:,2]

        loss_ret = self.mse(ret_pred, ret_t)
        loss_vol = self.mse(vol_pred, vol_t)
        loss_dir = self.bce(dir_pred, dir_t)

        return loss_ret + loss_vol + loss_dir



# ============================================================
#  TRAINING LOOP
# ============================================================

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
    dp_noise_std: float = 0.0,
):
    model.to(device)
    history = {"train_loss": [], "val_loss": []}

    if use_ewc:
        orig_params = {n: p.clone().detach() for n, p in model.named_parameters()}

    for epoch in range(1, epochs + 1):

        # ----------------------------
        # TRAIN
        # ----------------------------
        model.train()
        train_losses = []

        for batch in train_loader:
            if len(batch) == 2:
                X, y = batch
                extras = None
            else:
                X, y, chaos, resonance, regime,rqa = batch
                extras = (chaos, resonance, regime,rqa)

            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            if extras is not None:
                preds = model(
                    X,
                    extras[0].to(device),
                    extras[1].to(device),
                    extras[2].to(device),
                    extras[3].to(device),
                )
            else:
                preds = model(X)

            loss = criterion(preds, y)

            if use_ewc:
                ewc_penalty = 0.0
                for n, p in model.named_parameters():
                    ewc_penalty += ((p - orig_params[n]) ** 2).sum()
                loss += ewc_lambda * ewc_penalty
            if hasattr(model, "regime_embedding"):
                if epoch < 10:
                    model.regime_embedding.weight.requires_grad = False
                else:
                    model.regime_embedding.weight.requires_grad = True




            loss.backward()

            if dp_noise_std > 0.0:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad += torch.randn_like(p.grad) * dp_noise_std

            optimizer.step()
            train_losses.append(loss.item())

        # ----------------------------
        # VALIDATION
        # ----------------------------
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 2:
                    X, y = batch
                    extras = None
                else:
                    X, y, chaos, resonance, regime,rqa = batch
                    extras = (chaos, resonance, regime,rqa)

                X, y = X.to(device), y.to(device)

                if extras is not None:
                    preds = model(
                        X,
                        extras[0].to(device),
                        extras[1].to(device),
                        extras[2].to(device),
                        extras[3].to(device),

                    )
                else:
                    preds = model(X)

                val_losses.append(criterion(preds, y).item())

        history["train_loss"].append(float(np.mean(train_losses)))
        history["val_loss"].append(float(np.mean(val_losses)))

    return history


# ============================================================
#  GRID SEARCH
# ============================================================

def grid_search(
    model_class,
    param_grid: dict,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    epochs: int = 10,
    batch_size: int = 64,
):
    best_score = float("inf")
    best_params = None

    for params in ParameterGrid(param_grid):

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        val_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        model = model_class(
            **{
                k: params[k]
                for k in ["input_size", "hidden_size", "output_size", "num_layers"]
                if k in params
            }
        )

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params.get("lr", 1e-3))

        history = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            device,
            epochs=epochs,
        )

        score = min(history["val_loss"])

        if score < best_score:
            best_score = score
            best_params = params

    return best_params, best_score


# ============================================================
#  KERAS DEMO
# ============================================================

def run_keras_models(
    symbols: list,
    start_date: str,
    end_date: str,
    seq_len: int = 20,
    test_size: float = 0.2,
    epochs: int = 50,
    batch_size: int = 32,
    output_dir: str = "keras_runs",
):
    os.makedirs(output_dir, exist_ok=True)

    df = load_price_data(symbols, start_date, end_date)
    df = preprocess_data(df)
    df = compute_indicators(df)

    data = df["Close"].values.reshape(-1, 1)

    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=False
    )

    model = Sequential([
        LSTM(64, input_shape=(seq_len, 1)),
        Dropout(0.3),
        Dense(1),
    ])

    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")

    checkpoint_path = os.path.join(output_dir, "keras_prin_model.keras")
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True)
    early_stop_cb = EarlyStopping(patience=5, restore_best_weights=True)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop_cb, checkpoint_cb],
        verbose=2,
    )

    with open(os.path.join(output_dir, "history.json"), "w") as f:
        json.dump(history.history, f)


# ============================================================
#  PYTORCH PIPELINE — UPDATED TO PRIN++
# ============================================================

def run_pytorch_pipeline(
    symbols: list,
    start_date: str,
    end_date: str,
    seq_len: int = 20,
    test_size: float = 0.2,
    batch_size: int = 64,
    epochs: int = 50,
    device: str = "cpu",
    output_dir: str = "torch_runs",
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------
    df = load_price_data(symbols, start_date, end_date)
    df = preprocess_data(df)
    df = compute_indicators(df)

    # --------------------------------------------------------
    # BUILD PRIN++ SEQUENCES (THEOREM 1–5)
    # --------------------------------------------------------
    df_dict = {"data": df}

    (
        X_all,
        y_all,
        chaos_all,
        resonance_all,
        regimes_all,
        rqa_all
    ) = prepare_sequences_with_prin_plus_plus(
        df_dict,
        seq_length=seq_len,
        n_features=df.shape[1],
        n_regimes=3
    )

    # ===============================================
    # ENFORCE ALIGNMENT ACROSS ALL PRIN++ SEQUENCES
    # ===============================================
    N = min(
        len(X_all),
        len(y_all),
        len(chaos_all),
        len(resonance_all),
        len(regimes_all),
        len(rqa_all)
    )

    X_all         = X_all[:N]
    y_all         = y_all[:N]
    chaos_all     = chaos_all[:N]
    resonance_all = resonance_all[:N]
    regimes_all   = regimes_all[:N]
    rqa_all       = rqa_all[:N]



    # --------------------------------------------------------
    # TRAIN/VAL SPLIT
    # --------------------------------------------------------
     
    (
        X_train, X_val,
        y_train, y_val,
        chaos_train, chaos_val,
        resonance_train, resonance_val,
        regimes_train, regimes_val,
        rqa_train, rqa_val
    ) = train_test_split(
        X_all,
        y_all,
        chaos_all,
        resonance_all,
        regimes_all,
        rqa_all,
        test_size=test_size,
        random_state=42,
        shuffle=False,
    )

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------
    train_ds_prin = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_ds_prin = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )

    train_loader_prin = DataLoader(train_ds_prin, batch_size=batch_size, shuffle=True)
    val_loader_prin = DataLoader(val_ds_prin, batch_size=batch_size)

    train_ds_neuro = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(chaos_train, dtype=torch.float32),
        torch.tensor(resonance_train, dtype=torch.float32),
        torch.tensor(regimes_train, dtype=torch.long),
        torch.tensor(rqa_train, dtype=torch.float32),
    )

    val_ds_neuro = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
        torch.tensor(chaos_val, dtype=torch.float32),
        torch.tensor(resonance_val, dtype=torch.float32),
        torch.tensor(regimes_val, dtype=torch.long),
        torch.tensor(rqa_val, dtype=torch.float32),
    )


    train_loader_neuro = DataLoader(train_ds_neuro, batch_size=batch_size, shuffle=True)
    val_loader_neuro = DataLoader(val_ds_neuro, batch_size=batch_size)

    # --------------------------------------------------------
    # MODELS
    # --------------------------------------------------------
    feature_dim = X_all.shape[-1]

    prin_model = PRIN_LSTM(
        input_size=feature_dim,
        hidden_size=64,
        output_size=1,
        num_layers=1
    )

    base_model = BaselineLSTM(
        input_size=feature_dim,
        hidden_size=64,
        output_size=1
    )

    neuro_model = NeuroPRINv4(
        input_size=feature_dim,
        seq_len=seq_len,
        num_regimes=3,
        hidden_size=64,
        regime_embed_size=8,
        prune_rate=0.3,
    )

    criterion = MultiTaskLoss()


    optim_prin = torch.optim.Adam(prin_model.parameters(), lr=1e-3)
    optim_base = torch.optim.Adam(base_model.parameters(), lr=1e-3)
    optim_neuro = torch.optim.Adam(neuro_model.parameters(), lr=1e-3)

    # --------------------------------------------------------
    # TRAIN ALL THREE MODELS
    # --------------------------------------------------------
    hist_prin = train_model(prin_model, train_loader_prin, val_loader_prin, criterion, optim_prin, device, epochs)
    hist_base = train_model(base_model, train_loader_prin, val_loader_prin, criterion, optim_base, device, epochs)
    hist_neuro = train_model(neuro_model, train_loader_neuro, val_loader_neuro, criterion, optim_neuro, device, epochs)

    # --------------------------------------------------------
    # LATENCY MEASURES
    # --------------------------------------------------------
    lat_prin = measure_inference_latency(prin_model, torch.tensor(X_val[:100], dtype=torch.float32), device)
    lat_base = measure_inference_latency(base_model, torch.tensor(X_val[:100], dtype=torch.float32), device)

    neuro_model.eval()
    sample_X = torch.tensor(X_val[:100], dtype=torch.float32).to(device)
    sample_C = torch.tensor(chaos_val[:100], dtype=torch.float32).to(device)
    sample_R = torch.tensor(resonance_val[:100], dtype=torch.float32).to(device)
    sample_G = torch.tensor(regimes_val[:100], dtype=torch.long).to(device)

    sample_RQA = torch.tensor(rqa_val[:100], dtype=torch.float32).to(device)


    with torch.no_grad():
        for _ in range(10):
            _ = neuro_model(sample_X, sample_C, sample_R, sample_G, sample_RQA)

        times = []
        for _ in range(50):
            t0 = time.perf_counter()
            _ = neuro_model(sample_X, sample_C, sample_R, sample_G, sample_RQA)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)


    lat_neuro = float(np.mean(times))

    # --------------------------------------------------------
    # SAVE RUN METADATA
    # --------------------------------------------------------
    meta = {
        "prin_history": hist_prin,
        "base_history": hist_base,
        "neuro_history": hist_neuro,
        "prin_latency": lat_prin,
        "base_latency": lat_base,
        "neuro_latency": lat_neuro,
        "config": {
            "symbols": symbols,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "epochs": epochs,
            "test_size": test_size,
        },
    }

    save_run_metadata(os.path.join(output_dir, "run_meta.json"), meta)
    print("done")
