import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from hmmlearn.hmm import GaussianHMM

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton,
    QFileDialog, QLabel, QListWidget, QComboBox, QSpinBox, QDoubleSpinBox,
    QTabWidget, QTextEdit, QHBoxLayout, QTableWidget, QTableWidgetItem
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from neuroprin.data import (
    load_price_data,
    compute_indicators,
    prune_data,
    compute_fourier_resonance,
    compute_rqa_metrics,
    prepare_sequences_with_prin_plus_plus
)
from neuroprin.models import BaselineLSTM, PRIN_LSTM, DPRIN_LSTM, NeuroPRINv4
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


class NeuroPRINGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuroPRIN Studio Pro v4")
        self.df = None
        self.initUI()

    def initUI(self):
        widget = QWidget()
        layout = QVBoxLayout()

        self.file_label = QLabel("Select CSV Dataset:")
        layout.addWidget(self.file_label)
        self.load_button = QPushButton("Browse")
        self.load_button.clicked.connect(self.load_csv)
        layout.addWidget(self.load_button)

        self.feature_label = QLabel("Select Features:")
        layout.addWidget(self.feature_label)
        self.feature_list = QListWidget()
        self.feature_list.setSelectionMode(QListWidget.MultiSelection)
        layout.addWidget(self.feature_list)

        self.target_label = QLabel("Target:")
        layout.addWidget(self.target_label)
        self.target_box = QComboBox()
        layout.addWidget(self.target_box)

        self.model_label = QLabel("Model:")
        layout.addWidget(self.model_label)
        self.model_box = QComboBox()
        self.model_box.addItems(["BaselineLSTM", "PRIN_LSTM", "DPRIN_LSTM", "NeuroPRINv4"])
        layout.addWidget(self.model_box)

        self.epoch_label = QLabel("Epochs:")
        layout.addWidget(self.epoch_label)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setMaximum(100000)
        self.epochs_spin.setValue(50)
        layout.addWidget(self.epochs_spin)

        self.lr_label = QLabel("Learning Rate:")
        layout.addWidget(self.lr_label)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setValue(0.001)
        self.lr_spin.setDecimals(6)
        layout.addWidget(self.lr_spin)

        self.batch_label = QLabel("Batch Size:")
        layout.addWidget(self.batch_label)
        self.batch_spin = QSpinBox()
        self.batch_spin.setValue(64)
        layout.addWidget(self.batch_spin)

        self.seq_label = QLabel("Sequence Length:")
        layout.addWidget(self.seq_label)
        self.seq_spin = QSpinBox()
        self.seq_spin.setValue(10)
        layout.addWidget(self.seq_spin)

        self.regime_label = QLabel("HMM Regimes:")
        layout.addWidget(self.regime_label)
        self.regime_spin = QSpinBox()
        self.regime_spin.setValue(3)
        layout.addWidget(self.regime_spin)

        self.train_button = QPushButton("Run Training")
        self.train_button.clicked.connect(self.run_training)
        layout.addWidget(self.train_button)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self.loss_fig, self.loss_ax = plt.subplots()
        self.loss_canvas = FigureCanvas(self.loss_fig)
        self.tabs.addTab(self.loss_canvas, "Loss")

        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.tabs.addTab(self.summary_text, "Data Summary")

        self.heatmap_fig, self.heatmap_ax = plt.subplots()
        self.heatmap_canvas = FigureCanvas(self.heatmap_fig)
        self.tabs.addTab(self.heatmap_canvas, "Feature Correlation")

        # --- predictions panel with chart + table ---
        self.pred_widget = QWidget()
        self.pred_layout = QHBoxLayout(self.pred_widget)

        self.pred_fig, self.pred_ax = plt.subplots()
        self.pred_canvas = FigureCanvas(self.pred_fig)
        self.pred_layout.addWidget(self.pred_canvas)

        self.pred_table = QTableWidget()
        self.pred_layout.addWidget(self.pred_table)

        self.tabs.addTab(self.pred_widget, "Predictions")

        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select CSV File", "", "CSV Files (*.csv)")
        if not path: return
        self.df = load_price_data(path)
        self.df = compute_indicators(self.df)
        valid_cols = self.df.select_dtypes(include=[np.number, 'object']).columns.tolist()
        self.feature_list.clear()
        self.feature_list.addItems(valid_cols)
        self.target_box.clear()
        self.target_box.addItems(valid_cols)
        self.display_summary()
        self.display_correlation()

    def display_summary(self):
        summary = self.df.describe(include='all').T
        summary['Missing'] = self.df.isna().sum()
        self.summary_text.setText(str(summary))

    def display_correlation(self):
        num_df = self.df.select_dtypes(include=[np.number])
        self.heatmap_ax.clear()
        sns.heatmap(num_df.corr(), ax=self.heatmap_ax, cmap='coolwarm', annot=False)
        self.heatmap_canvas.draw()

    def run_training(self):
        selected = [i.text() for i in self.feature_list.selectedItems()]
        target = self.target_box.currentText()
        if not selected or not target: return
        df_train = self.df[selected + [target]].dropna()

        model_type = self.model_box.currentText()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        hidden_size = 64
        output_size = 1

        if model_type == 'NeuroPRINv4':
            seq_len = self.seq_spin.value()
            n_regimes = self.regime_spin.value()

            df_p = prune_data(df_train)
            df_r = compute_fourier_resonance(df_p, window_size=seq_len)
            df_q = compute_rqa_metrics(df_r, window_size=seq_len)

            X, y, chaos_arr, rec_arr, regimes_arr, _ = prepare_sequences_with_prin_plus_plus(
                {'_': df_q}, seq_length=seq_len, n_features=len(selected), n_regimes=n_regimes
            )

            split = int(len(X) * 0.8)
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]
            chaos_train, chaos_val = chaos_arr[:split], chaos_arr[split:]
            reg_train, reg_val = regimes_arr[:split], regimes_arr[split:]

            y_scaler = StandardScaler()
            y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))
            y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1))

            model = NeuroPRINv4(input_size=X.shape[-1], seq_len=seq_len, num_regimes=n_regimes)
            X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
            X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
            y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)
            y_val_t = torch.tensor(y_val_scaled, dtype=torch.float32).to(device)
            chaos_train_t = torch.tensor(chaos_train, dtype=torch.float32).to(device)
            chaos_val_t = torch.tensor(chaos_val, dtype=torch.float32).to(device)
            reg_train_t = torch.tensor(reg_train, dtype=torch.long).to(device)
            reg_val_t = torch.tensor(reg_val, dtype=torch.long).to(device)

            optimizer = optim.Adam(model.parameters(), lr=self.lr_spin.value())
            criterion = nn.MSELoss()

            history = {'loss': [], 'val_loss': []}
            for ep in range(self.epochs_spin.value()):
                model.train()
                optimizer.zero_grad()
                out = model(X_train_t, chaos_train_t, reg_train_t)
                loss = criterion(out, y_train_t)
                loss.backward()
                optimizer.step()
                model.eval()
                with torch.no_grad():
                    val_out = model(X_val_t, chaos_val_t, reg_val_t)
                    val_loss = criterion(val_out, y_val_t)
                history['loss'].append(loss.item())
                history['val_loss'].append(val_loss.item())

            self.plot_loss(history)
            self.plot_predictions(model, X_val_t, y_val_t, chaos_val_t, reg_val_t, scaler_y=y_scaler)
            return

        # Standard models:
        X = pd.get_dummies(df_train[selected], drop_first=True).values.astype(np.float32)
        y = df_train[target].values.astype(np.float32).reshape(-1, 1)
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        input_size = X.shape[1]
        model = {
            'BaselineLSTM': BaselineLSTM,
            'PRIN_LSTM': PRIN_LSTM,
            'DPRIN_LSTM': DPRIN_LSTM
        }[model_type](input_size, hidden_size, output_size)
        model.to(device)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        y_scaler = StandardScaler()
        y_train = y_scaler.fit_transform(y_train)
        y_val = y_scaler.transform(y_val)

        X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1).to(device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

        optimizer = optim.Adam(model.parameters(), lr=self.lr_spin.value())
        criterion = nn.MSELoss()

        history = {'loss': [], 'val_loss': []}
        for ep in range(self.epochs_spin.value()):
            model.train()
            optimizer.zero_grad()
            out = model(X_train_t)
            loss = criterion(out, y_train_t)
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_val_t), y_val_t)
            history['loss'].append(loss.item())
            history['val_loss'].append(val_loss.item())

        self.plot_loss(history)
        self.plot_predictions(model, X_val_t, y_val_t, scaler_y=y_scaler)

    def plot_loss(self, history):
        self.loss_ax.clear()
        self.loss_ax.plot(history['loss'], label='Train')
        self.loss_ax.plot(history['val_loss'], label='Val')
        self.loss_ax.legend()
        self.loss_ax.set_title('Training Loss')
        self.loss_fig.tight_layout()
        self.loss_canvas.draw()

    def plot_predictions(self, model, X_val, y_val, chaos=None, regime=None, scaler_y=None):
        model.eval()
        with torch.no_grad():
            if hasattr(model, 'forward') and chaos is not None:
                y_pred = model(X_val, chaos, regime).cpu().numpy()
            else:
                y_pred = model(X_val).cpu().numpy()

        y_pred_rescaled = scaler_y.inverse_transform(y_pred)
        y_val_rescaled = scaler_y.inverse_transform(y_val.cpu().numpy())

        mae = mean_absolute_error(y_val_rescaled, y_pred_rescaled)
        rmse = mean_squared_error(y_val_rescaled, y_pred_rescaled, squared=False)
        r2 = r2_score(y_val_rescaled, y_pred_rescaled)

        self.pred_ax.clear()
        self.pred_ax.plot(y_val_rescaled, label='Actual')
        self.pred_ax.plot(y_pred_rescaled, label='Predicted')
        self.pred_ax.set_title(f'Predictions (MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.3f})')
        self.pred_ax.legend()
        self.pred_fig.tight_layout()
        self.pred_canvas.draw()

        self.pred_table.clear()
        self.pred_table.setRowCount(len(y_val_rescaled))
        self.pred_table.setColumnCount(3)
        self.pred_table.setHorizontalHeaderLabels(["Index", "Actual", "Predicted"])

        for i in range(len(y_val_rescaled)):
            self.pred_table.setItem(i, 0, QTableWidgetItem(str(i)))
            self.pred_table.setItem(i, 1, QTableWidgetItem(f"{y_val_rescaled[i][0]:.3f}"))
            self.pred_table.setItem(i, 2, QTableWidgetItem(f"{y_pred_rescaled[i][0]:.3f}"))

        self.pred_table.resizeColumnsToContents()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = NeuroPRINGUI()
    window.show()
    sys.exit(app.exec_())
