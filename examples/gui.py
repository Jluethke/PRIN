# ======================================================================
#  NeuroPRIN Studio Pro v5 — Full Option-E Stability Patch
# ======================================================================

import sys
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List
import torch
import torch.nn as nn

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton,
    QFileDialog, QLabel, QListWidget, QComboBox, QSpinBox,
    QDoubleSpinBox, QTabWidget, QTextEdit, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QDialog, QProgressBar
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# === PRIN / NEUROPRIN IMPORTS ===
from neuroprin.data import (
    load_price_data, compute_indicators, prepare_sequences_with_prin_plus_plus
)
from neuroprin.models import (
    BaselineLSTM, PRIN_LSTM, DPRIN_LSTM, NeuroPRINv4,
    LinearRegressionModel, RandomForestModel, XGBoostModel,
    SupportVectorModel, KNNModel, GradientBoostingModel
)
from Empirical_Output import run_all_empirical_tests

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# ======================================================================
#  EMPIRICAL DIAGNOSTICS POPUP WINDOW
# ======================================================================

class EmpiricalDiagnosticsWindow(QDialog):
    def __init__(self, parent=None, empirical_data=None):
        super().__init__(parent)

        self.setModal(True)
        self.setWindowTitle("PRIN Empirical Diagnostics Suite")
        self.resize(900, 600)

        self.empirical_data = empirical_data if empirical_data else {}

        layout = QVBoxLayout()

        # Button
        self.run_all_button = QPushButton("Generate Full Empirical PDF Bundle")
        self.run_all_button.clicked.connect(self.run_full_suite)
        layout.addWidget(self.run_all_button)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.progress)

        # Log output
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        layout.addWidget(self.log_box)

        self.setLayout(layout)

    def _log(self, msg):
        self.log_box.append(msg)
        QApplication.processEvents()

    def _step(self, pct):
        self.progress.setValue(pct)
        QApplication.processEvents()

    def run_full_suite(self):

        if not self.empirical_data:
            self._log("[ERROR] No empirical data. Train NeuroPRINv4 first.")
            return

        self.run_all_button.setEnabled(False)
        self._step(0)
        self._log("[INFO] Running empirical diagnostics...")

        try:
            d = self.empirical_data

            run_all_empirical_tests(
                model=d["model"],
                Xv=d["Xv"],
                chaos=d["chaos"],
                resonance=d["resonance"],
                regimes=d["regimes"],
                rqa=d["rqa"],
                y_true=d["y_true"],
                y_pred=d["y_pred"]
            )

            self._step(100)
            self._log("[SUCCESS] Diagnostics written to ./empirical/")

        except Exception as e:
            self._log(f"[ERROR] {e}")

        finally:
            self.run_all_button.setEnabled(True)


# ======================================================================
#  MAIN GUI
# ======================================================================

class NeuroPRINGUI(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("NeuroPRIN Studio Pro v5 — Stable Edition")

        self.df = None
        self.empirical_data = {}

        # GUI init
        self.initUI()



    # ==================================================================
    #  BUILD UI
    # ==================================================================
    def initUI(self):

        widget = QWidget()
        layout = QVBoxLayout()

        # --------------------------------------------------------------
        # CSV Loader
        # --------------------------------------------------------------
        self.load_button = QPushButton("Browse CSV")
        self.load_button.clicked.connect(self.load_csv)
        layout.addWidget(self.load_button)

        # --------------------------------------------------------------
        # Feature selection
        # --------------------------------------------------------------
        layout.addWidget(QLabel("Features:"))
        self.feature_list = QListWidget()
        self.feature_list.setSelectionMode(QListWidget.MultiSelection)
        layout.addWidget(self.feature_list)

        # --------------------------------------------------------------
        # Target selection
        # --------------------------------------------------------------
        layout.addWidget(QLabel("Target:"))
        self.target_box = QComboBox()
        layout.addWidget(self.target_box)

        # --------------------------------------------------------------
        # Model selection
        # --------------------------------------------------------------
        self.model_box = QComboBox()
        self.model_box.addItems([
            "BaselineLSTM", "DPRIN_LSTM", "PRIN_LSTM",
            "NeuroPRINv4",
            "LinearRegression", "RandomForest", "XGBoost",
            "SVR", "KNN", "GradientBoosting"
        ])
        layout.addWidget(QLabel("Model:"))
        layout.addWidget(self.model_box)

        # --------------------------------------------------------------
        # Hyperparameters
        # --------------------------------------------------------------
        layout.addWidget(QLabel("Epochs"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 20000)
        self.epochs_spin.setValue(200)
        layout.addWidget(self.epochs_spin)

        layout.addWidget(QLabel("Learning Rate"))
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(6)
        self.lr_spin.setValue(1e-3)
        layout.addWidget(self.lr_spin)

        layout.addWidget(QLabel("Sequence Length"))
        self.seq_spin = QSpinBox()
        self.seq_spin.setRange(2, 500)
        self.seq_spin.setValue(20)
        layout.addWidget(self.seq_spin)

        layout.addWidget(QLabel("HMM Regimes"))
        self.regime_spin = QSpinBox()
        self.regime_spin.setRange(1, 12)
        self.regime_spin.setValue(3)
        layout.addWidget(self.regime_spin)

        layout.addWidget(QLabel("Forecast Steps (NeuroPRINv4 only)"))
        self.forecast_spin = QSpinBox()
        self.forecast_spin.setRange(1, 100)
        self.forecast_spin.setValue(10)
        layout.addWidget(self.forecast_spin)

        # --------------------------------------------------------------
        # Buttons
        # --------------------------------------------------------------
        self.train_button = QPushButton("Run Training")
        self.train_button.clicked.connect(self.run_training)
        layout.addWidget(self.train_button)

        self.empirical_button = QPushButton("Run Empirical Validation Suite")
        self.empirical_button.clicked.connect(self.launch_empirical_suite)
        layout.addWidget(self.empirical_button)

        # --------------------------------------------------------------
        # Tabs
        # --------------------------------------------------------------
        self.tabs = QTabWidget()

        # Loss tab
        self.loss_fig, self.loss_ax = plt.subplots()
        self.loss_canvas = FigureCanvas(self.loss_fig)
        self.tabs.addTab(self.loss_canvas, "Loss")

        # Summary tab
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.tabs.addTab(self.summary_text, "Summary")

        # Correlation tab
        self.heatmap_fig, self.heatmap_ax = plt.subplots()
        self.heatmap_canvas = FigureCanvas(self.heatmap_fig)
        self.tabs.addTab(self.heatmap_canvas, "Correlation")

        # Predictions tab
        pred_widget = QWidget()
        pred_layout = QHBoxLayout(pred_widget)

        self.pred_fig, self.pred_ax = plt.subplots()
        self.pred_canvas = FigureCanvas(self.pred_fig)
        pred_layout.addWidget(self.pred_canvas)

        self.pred_table = QTableWidget()
        pred_layout.addWidget(self.pred_table)

        self.tabs.addTab(pred_widget, "Predictions")

        layout.addWidget(self.tabs)
        widget.setLayout(layout)
        self.setCentralWidget(widget)



    # ======================================================================
    #  LOAD CSV
    # ======================================================================
    def load_csv(self):

        path, _ = QFileDialog.getOpenFileName(
            self, "Select CSV File", "", "CSV Files (*.csv)"
        )
        if not path:
            return

        # Load + indicators
        self.df = load_price_data(path)
        self.df = compute_indicators(self.df)

        # numeric features only
        cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        self.feature_list.clear()
        self.feature_list.addItems(cols)

        self.target_box.clear()
        self.target_box.addItems(cols)

        # summary
        summary = self.df.describe().T
        summary["Missing"] = self.df.isna().sum()
        self.summary_text.setText(str(summary))

        # correlation heatmap
        self.heatmap_ax.clear()
        sns.heatmap(self.df[cols].corr(), cmap="coolwarm", ax=self.heatmap_ax)
        self.heatmap_canvas.draw()


    # ======================================================================
    #  PLOT LOSS CURVE
    # ======================================================================
    def plot_loss(self, hist):
        self.loss_ax.clear()
        self.loss_ax.plot(hist["loss"], label="Train")
        self.loss_ax.plot(hist["val"], label="Validation")
        self.loss_ax.legend()
        self.loss_ax.set_title("Loss Over Epochs")
        self.loss_canvas.draw()


    # ======================================================================
    #  PLOT PREDICTIONS
    # ======================================================================
    def plot_preds_from_numpy(
        self,
        dates,
        yv,
        yp,
        forecast=None,
        forecast_dates=None,
        title_prefix=""
    ):
        self.pred_ax.clear()

        yv = np.array(yv).reshape(-1)
        yp = np.array(yp).reshape(-1)

        # Compute metrics safely
        if len(yv) == len(yp):
            mae = mean_absolute_error(yv, yp)
            rmse = mean_squared_error(yv, yp, squared=False)
            r2 = r2_score(yv, yp)
            title = f"{title_prefix} MAE={mae:.4f} RMSE={rmse:.4f} R²={r2:.4f}"
        else:
            title = title_prefix

        # Plot actual vs predicted
        self.pred_ax.plot(dates, yv, label="Actual")
        self.pred_ax.plot(dates, yp, label="Predicted")

        # Optional forecasting line
        if forecast is not None:
            self.pred_ax.plot(
                forecast_dates,
                np.array(forecast).reshape(-1),
                linestyle="--",
                label="Forecast"
            )
            self.pred_ax.axvline(dates[-1], color="gray", linestyle=":")

        self.pred_ax.set_title(title)
        self.pred_ax.legend()
        self.pred_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        self.pred_fig.autofmt_xdate()
        self.pred_canvas.draw()

        # Table
        self.pred_table.clear()
        self.pred_table.setRowCount(len(yv))
        self.pred_table.setColumnCount(3)
        self.pred_table.setHorizontalHeaderLabels(["Index", "Actual", "Predicted"])

        for i, (a, p) in enumerate(zip(yv, yp)):
            self.pred_table.setItem(i, 0, QTableWidgetItem(str(i)))
            self.pred_table.setItem(i, 1, QTableWidgetItem(f"{a:.4f}"))
            self.pred_table.setItem(i, 2, QTableWidgetItem(f"{p:.4f}"))

        self.pred_table.resizeColumnsToContents()
        self.tabs.setCurrentIndex(3)


    # ======================================================================
    #  TRAINING MAIN FUNCTION
    # ======================================================================
    def run_training(self):

        # Basic checks
        if self.df is None:
            self.summary_text.setText("[ERROR] Load CSV first.")
            return

        sel = [i.text() for i in self.feature_list.selectedItems()]
        tgt = self.target_box.currentText()
        model_type = self.model_box.currentText()

        if not sel or not tgt:
            self.summary_text.setText("[ERROR] Select features and target.")
            return

        df = self.df[sel + [tgt]].copy()   # KEEP THIS FOR SIMPLE LSTM & ML
        df_features_only = df[sel].copy()  # USE THIS FOR PRIN++

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill().bfill()


        if len(df) < 50:
            self.summary_text.setText("[ERROR] Not enough rows after cleaning.")
            return

        L = self.seq_spin.value()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        forecast_steps = self.forecast_spin.value()

        # ==================================================================
        # 1) Classical ML Models — NO SEQUENCES
        # ==================================================================
        if model_type in [
            "LinearRegression", "RandomForest", "XGBoost",
            "SVR", "KNN", "GradientBoosting"
        ]:
            try:
                X = df[sel].values
                y = df[tgt].values

                split = int(len(X) * 0.8)
                X_train, X_test = X[:split], X[split:]
                y_train, y_test = y[:split], y[split:]

                model_map = {
                    "LinearRegression": LinearRegressionModel(len(sel), 1),
                    "RandomForest": RandomForestModel(),
                    "XGBoost": XGBoostModel(),
                    "SVR": SupportVectorModel(),
                    "KNN": KNNModel(),
                    "GradientBoosting": GradientBoostingModel(),
                }

                model = model_map[model_type]
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                val_idx = df.index[-len(y_test):]
                self.plot_preds_from_numpy(
                    val_idx, y_test, preds, title_prefix=f"{model_type} — "
                )

                self.summary_text.setText(f"{model_type} Training Complete.")
                return

            except Exception as e:
                self.summary_text.setText(f"[ERROR classical model]\n{e}")
                return


        # ==================================================================
        # 2) SIMPLE LSTM MODELS — Baseline / DPRIN / PRIN
        # ==================================================================
        if model_type in ["BaselineLSTM", "DPRIN_LSTM", "PRIN_LSTM"]:
            try:

                vals = df.values
                if len(vals) < L + 5:
                    self.summary_text.setText("[ERROR] Not enough rows for LSTM.")
                    return

                # Sliding window creation
                X_seq, y_seq = [], []
                for i in range(L, len(vals)):
                    target_idx = df.columns.get_loc(tgt)
                    X_seq.append(np.delete(vals[i - L:i], target_idx, axis=1))
                    y_seq.append(vals[i, target_idx])


                X_seq = np.array(X_seq, dtype=np.float32)
                y_seq = np.array(y_seq, dtype=np.float32)

                # Normalize input windows
                scaler_X = StandardScaler()
                X_seq = scaler_X.fit_transform(
                    X_seq.reshape(-1, X_seq.shape[-1])
                ).reshape(X_seq.shape)

                scaler_y = StandardScaler()
                y_seq = scaler_y.fit_transform(y_seq.reshape(-1, 1)).reshape(-1)

                # Torch tensors
                X_seq = torch.tensor(X_seq, dtype=torch.float32).to(device)
                y_seq = torch.tensor(y_seq, dtype=torch.float32).to(device)

                split = int(len(X_seq) * 0.8)
                Xt, Xv = X_seq[:split], X_seq[split:]
                yt, yv = y_seq[:split], y_seq[split:]

                # Model definitions
                model_map = {
                    "BaselineLSTM": BaselineLSTM(len(sel), 64, 1, num_layers=1, dropout_p=0.0),
                    "DPRIN_LSTM": DPRIN_LSTM(len(sel), 64, 1),
                    "PRIN_LSTM": PRIN_LSTM(len(sel), 64, 1, num_layers=1,
                                            pruning_threshold=0.001, dropout_p=0.0),
                }
                model = model_map[model_type].to(device)

                opt = torch.optim.Adam(model.parameters(), lr=self.lr_spin.value())
                loss_fn = nn.MSELoss()

                hist = {"loss": [], "val": []}

                # Patched training: gradient clipping + stable LR
                for epoch in range(self.epochs_spin.value()):
                    model.train()
                    opt.zero_grad()

                    pred = model(Xt).squeeze()
                    loss = loss_fn(pred, yt)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()

                    model.eval()
                    with torch.no_grad():
                        vpred = model(Xv).squeeze()
                        vloss = loss_fn(vpred, yv)

                    hist["loss"].append(loss.item())
                    hist["val"].append(vloss.item())

                    self.plot_loss(hist)

                # Undo scaling
                yv_np = scaler_y.inverse_transform(yv.cpu().numpy().reshape(-1, 1)).reshape(-1)
                yp_np = scaler_y.inverse_transform(vpred.cpu().numpy().reshape(-1, 1)).reshape(-1)

                val_idx = df.index[-len(yv_np):]

                self.plot_preds_from_numpy(
                    val_idx, yv_np, yp_np, title_prefix=f"{model_type} — "
                )

                self.summary_text.setText(f"{model_type} Training Complete.")
                return

            except Exception as e:
                self.summary_text.setText(f"[ERROR in LSTM]\n{e}")
                return

        # ==================================================================
        # 3) NEUROPRINv4 — FULL CHAOS / RESONANCE / RQA / HMM / P-CODING
        # ==================================================================
        if model_type == "NeuroPRINv4":
            try:
                L = self.seq_spin.value()
                R = self.regime_spin.value()
                out = prepare_sequences_with_prin_plus_plus(
                    {"data": df},
                    seq_length=L,
                    n_features=len(sel),
                    n_regimes=R
                )






                X         = out["X"]
                y         = out["y"]
                chaos     = out["chaos"]
                resonance = out["resonance"]
                regimes   = out["regimes"]
                rqa       = out["rqa"]

                # ===========================================================
                # ALIGNMENT AFTER OUTPUT EXTRACTION — CORRECT LOCATION
                # ===========================================================
                N = min(len(X), len(y), len(chaos), len(resonance), len(regimes), len(rqa))

                X = X[:N]
                y = y[:N]
                chaos = chaos[:N]
                resonance = resonance[:N]
                regimes = regimes[:N]
                rqa = rqa[:N]

                print(f"[DEBUG] Correct alignment applied. Final N={N}")


                # Convert numpy arrays to tensors
                X = torch.tensor(X, dtype=torch.float32)
                y = torch.tensor(y, dtype=torch.float32)
                chaos = torch.tensor(chaos, dtype=torch.float32)
                resonance = torch.tensor(resonance, dtype=torch.float32)
                regimes = torch.tensor(regimes, dtype=torch.long)
                rqa = torch.tensor(rqa, dtype=torch.float32)

                # Debugging confirmation
                print("\n=== DEBUG NeuroPRINv4 TENSOR SHAPES ===")
                print("X:", X.shape)
                print("y:", y.shape)
                print("chaos:", chaos.shape)
                print("resonance:", resonance.shape)
                print("regimes:", regimes.shape)
                print("rqa:", rqa.shape)
                print("=======================================\n")



                if len(X) < 40:
                    self.summary_text.setText("[ERROR] Not enough sequences for NeuroPRINv4.")
                    return
                split = int(len(X) * 0.8)


                # Split main sequences
                Xt, Xv = X[:split], X[split:]
                yt, yv = y[:split], y[split:]

                # Split PRIN++ auxiliary sequences
                ct,  cv  = chaos[:split],     chaos[split:]
                rst, rsv = resonance[:split], resonance[split:]
                rt,  rv  = regimes[:split],   regimes[split:]
                rqa_t    = rqa[:split]
                rqa_v    = rqa[split:]

                # FULL RAW SEQUENCES (no pooling) → tensors
                ct_raw  = torch.tensor(ct,     dtype=torch.float32).to(device)   # [N, L]
                cv_raw  = torch.tensor(cv,     dtype=torch.float32).to(device)   # [M, L]
                rst_raw = torch.tensor(rst,    dtype=torch.float32).to(device)   # [N, L]
                rsv_raw = torch.tensor(rsv,    dtype=torch.float32).to(device)   # [M, L]
                rt_raw  = torch.tensor(rt,     dtype=torch.long).to(device)      # [N, L]
                rv_raw  = torch.tensor(rv,     dtype=torch.long).to(device)      # [M, L]

                # Convert X and y windows to tensors
                # Convert X and y windows to tensors
                Xt_t = torch.tensor(Xt, dtype=torch.float32).to(device)
                Xv_t = torch.tensor(Xv, dtype=torch.float32).to(device)

                # y is already [N, 3] from PRIN++ (ret, dir, vol)
                yt_t = torch.tensor(yt, dtype=torch.float32).to(device)   # [N_train, 3]
                yv_t = torch.tensor(yv, dtype=torch.float32).to(device)   # [N_val, 3]


                # RQA sequences to tensors
                rqa_raw   = torch.tensor(rqa_t, dtype=torch.float32).to(device)  # [N, L, 10]
                rqa_v_raw = torch.tensor(rqa_v, dtype=torch.float32).to(device)  # [M, L, 10]

                # ============================================
                # HARD ALIGNMENT CHECK — PREVENTS CRASH
                # ============================================
                train_lengths = [
                    Xt_t.shape[0],
                    ct_raw.shape[0],
                    rst_raw.shape[0],
                    rt_raw.shape[0],
                    rqa_raw.shape[0],
                ]

                if len(set(train_lengths)) != 1:
                    raise ValueError(
                        f"PRIN++ SEQUENCE LENGTH MISMATCH:\n"
                        f"Xt={Xt_t.shape}, chaos={ct_raw.shape}, resonance={rst_raw.shape}, "
                        f"regimes={rt_raw.shape}, rqa={rqa_raw.shape}\n\n"
                        f"This mismatch would cause the tensor error "
                        f"'size of tensor a ({train_lengths[0]}) must match size of tensor b "
                        f"({train_lengths[1]}) at non-singleton dimension 0'."
                    )

                #====================
                #      Debug       #
                
                #==================#


                print("\n===== DEBUG SHAPES =====")

                print("Features selected (sel):", sel)
                print("Number of features (len(sel)):", len(sel))

                print("\nX shapes:")
                print("X.shape =", X.shape)
                print("Xt.shape =", Xt.shape)
                print("Xv.shape =", Xv.shape)

                print("\nTorch tensor shapes:")
                print("Xt_t.shape =", Xt_t.shape, " last-dim =", Xt_t.shape[-1])
                print("Xv_t.shape =", Xv_t.shape, " last-dim =", Xv_t.shape[-1])

                print("\nChaos shapes:")
                print("chaos.shape =", chaos.shape)
                print("ct_raw.shape =", ct_raw.shape)

                print("\nResonance shapes:")
                print("resonance.shape =", resonance.shape)
                print("rst_raw.shape =", rst_raw.shape)

                print("\nRegime shapes:")
                print("regimes.shape =", regimes.shape)
                print("rt_raw.shape =", rt_raw.shape)

                print("\nRQA shapes:")
                print("rqa.shape =", rqa.shape)
                print("rqa_raw.shape =", rqa_raw.shape)

                # Defensive trap: crash ALREADY if mismatch exists
                if Xt_t.shape[-1] != len(sel):
                    print("\n\n============================")
                    print("FATAL FEATURE MISMATCH FOUND")
                    print("============================")
                    print(f"Expected features (len(sel)): {len(sel)}")
                    print(f"Actual features in X: {Xt_t.shape[-1]}")
                    print("\nThis means the target column IS STILL in X.")
                    print("STOPPING NOW...")
                    raise ValueError("Dimensionality mismatch: Target still included in X")

                print("===== DEBUG COMPLETE =====\n")


                # Create model
                model = NeuroPRINv4(
                    input_size=X.shape[-1],
                    seq_len=L,
                    num_regimes=R
                ).to(device)

                opt = torch.optim.Adam(model.parameters(), lr=self.lr_spin.value())
                loss_fn = nn.MSELoss()

                hist = {"loss": [], "val": []}

                # TRAINING LOOP (patched for stability)
                for epoch in range(self.epochs_spin.value()):
                    model.train()
                    opt.zero_grad()

                    pred = model(Xt_t, ct_raw, rst_raw, rt_raw,rqa_raw)
                    print("[DEBUG] pred.shape:", pred.shape, "yt_t.shape:", yt_t.shape)

                    loss = loss_fn(pred, yt_t)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()

                    model.eval()
                    with torch.no_grad():
                        vpred = model(Xv_t, cv_raw, rsv_raw, rv_raw,rqa_v_raw)

                        vloss = loss_fn(vpred, yv_t)

                    hist["loss"].append(loss.item())
                    hist["val"].append(vloss.item())
                    self.plot_loss(hist)

                # Convert predictions
                yv_true = yv.reshape(-1)
                yp_val = vpred.detach().cpu().numpy().reshape(-1)

                # ============================
                #  FORECASTING (N-step)
                # ============================
                forecast = []

                # Start from last validation window
                last_X      = Xv_t[-1:].clone()        # [1, L, F]
                last_c      = cv_raw[-1:].clone()      # [1, L]
                last_r      = rsv_raw[-1:].clone()     # [1, L]
                last_regime = rv_raw[-1:].clone()      # [1, L]
                last_rqa = rqa_v_raw[-1:].clone()


                with torch.no_grad():
                    for _ in range(forecast_steps):

                        # Run model to get next prediction vector [1, 3]
                        nxt = model(last_X, last_c, last_r, last_regime, last_rqa)
                        nxt_val = nxt[0, 0].item()


                        # Extract NEXT RETURN ONLY (scalar)
                        nxt_val = nxt[0, 0].item()
                        forecast.append(nxt_val)

                        # Build new feature step
                        new_step = last_X[:, -1:, :].clone()
                        new_step[..., 0] = nxt_val  # overwrite 'Close/Return' channel

                        # Shift windows
                        last_X      = torch.cat([last_X[:, 1:], new_step], dim=1)
                        last_c      = torch.cat([last_c[:, 1:], last_c[:, -1:].clone()], dim=1)
                        last_r      = torch.cat([last_r[:, 1:], last_r[:, -1:].clone()], dim=1)
                        last_regime = torch.cat([last_regime[:, 1:], last_regime[:, -1:].clone()], dim=1)
                        last_rqa    = torch.cat([last_rqa[:, :, 1:], last_rqa[:, :, -1:].clone()], dim=2)


                # Build forecast date range
                delta = df.index[-1] - df.index[-2]
                forecast_dates = [
                    df.index[-1] + (i + 1) * delta for i in range(len(forecast))
                ]

                # Plot validation region
                val_idx = df.index[-len(yv_true):]
                self.plot_preds_from_numpy(
                    val_idx,
                    yv_true,
                    yp_val,
                    forecast=forecast,
                    forecast_dates=forecast_dates,
                    title_prefix="NeuroPRINv4 — "
                )

                self.summary_text.setText("NeuroPRINv4 Training Complete.")

                # ============================
                #  Store empirical diagnostics
                # ============================
                raw_chaos = chaos.clone().cpu().numpy()
                raw_resonance = resonance.clone().cpu().numpy()
                raw_regimes = regimes.clone().cpu().numpy()
                raw_rqa = rqa.clone().cpu().numpy()
                raw_Xv = Xv.clone().cpu().numpy()


                self.empirical_data = {
                    "model": model,
                    "Xv": raw_Xv,
                    "chaos": raw_chaos,
                    "resonance": raw_resonance,
                    "regimes": raw_regimes,
                    "rqa": raw_rqa,
                    "y_true": yv_true,
                    "y_pred": yp_val,
                }

                return

            except Exception as e:
                self.summary_text.setText(f"[ERROR NeuroPRINv4]\n{e}")
                return


    # ======================================================================
    #  EMPIRICAL SUITE LAUNCHER
    # ======================================================================
    def launch_empirical_suite(self):
        print("[DEBUG] empirical_data keys:", self.empirical_data.keys())

        if not self.empirical_data:
            self.summary_text.setText(
                "[ERROR] Run NeuroPRINv4 training first."
            )
            return

        try:
            win = EmpiricalDiagnosticsWindow(
                parent=self,
                empirical_data=self.empirical_data
            )
            win.exec_()

        except Exception as e:
            self.summary_text.setText(f"[ERROR launching diagnostics]: {e}")


# ======================================================================
#  MAIN ENTRY POINT
# ======================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = NeuroPRINGUI()
    win.show()
    sys.exit(app.exec_())
