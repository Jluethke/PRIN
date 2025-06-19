import sys
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List 
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
from neuroprin.models import (
    BaselineLSTM, PRIN_LSTM, DPRIN_LSTM, NeuroPRINv4,
    LinearRegressionModel, RandomForestModel, XGBoostModel,
    SupportVectorModel, KNNModel, GradientBoostingModel
)
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

        self.load_button = QPushButton("Browse CSV")
        self.load_button.clicked.connect(self.load_csv)
        layout.addWidget(self.load_button)

        self.feature_list = QListWidget()
        self.feature_list.setSelectionMode(QListWidget.MultiSelection)
        layout.addWidget(QLabel("Features:"))
        layout.addWidget(self.feature_list)

        self.target_box = QComboBox()
        layout.addWidget(QLabel("Target:"))
        layout.addWidget(self.target_box)

        self.model_box = QComboBox()
        self.model_box.addItems([
            "BaselineLSTM", "DPRIN_LSTM", "NeuroPRINv4",
            "LinearRegression", "RandomForest", "XGBoost", "SVR", "KNN", "GradientBoosting"
        ])
        layout.addWidget(QLabel("Model:"))
        layout.addWidget(self.model_box)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setMinimum(1)
        self.epochs_spin.setMaximum(100000)
        self.epochs_spin.setValue(500)
        layout.addWidget(QLabel("Epochs:"))
        layout.addWidget(self.epochs_spin)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(6)
        self.lr_spin.setValue(0.001)
        layout.addWidget(QLabel("Learning Rate:"))
        layout.addWidget(self.lr_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setValue(64)
        layout.addWidget(QLabel("Batch Size:"))
        layout.addWidget(self.batch_spin)

        self.seq_spin = QSpinBox()
        self.seq_spin.setValue(10)
        layout.addWidget(QLabel("Sequence Length:"))
        layout.addWidget(self.seq_spin)

        self.regime_spin = QSpinBox()
        self.regime_spin.setValue(3)
        layout.addWidget(QLabel("HMM Regimes:"))
        layout.addWidget(self.regime_spin)

        self.forecast_spin = QSpinBox()
        self.forecast_spin.setRange(1, 100)
        self.forecast_spin.setValue(5)
        layout.addWidget(QLabel("Forecast Steps:"))
        layout.addWidget(self.forecast_spin)


        self.train_button = QPushButton("Run Training")
        self.train_button.clicked.connect(self.run_training)
        layout.addWidget(self.train_button)

        self.tabs = QTabWidget()
        self.loss_fig, self.loss_ax = plt.subplots()
        self.loss_canvas = FigureCanvas(self.loss_fig)
        self.tabs.addTab(self.loss_canvas, "Loss")

        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.tabs.addTab(self.summary_text, "Summary")

        self.heatmap_fig, self.heatmap_ax = plt.subplots()
        self.heatmap_canvas = FigureCanvas(self.heatmap_fig)
        self.tabs.addTab(self.heatmap_canvas, "Correlation")

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

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select CSV", "", "CSV Files (*.csv)")
        if not path:
            return
        self.df = load_price_data(path)
        self.df = compute_indicators(self.df)
        cols = self.df.select_dtypes(include=[np.number, 'object']).columns.tolist()
        self.feature_list.clear()
        self.feature_list.addItems(cols)
        self.target_box.clear()
        self.target_box.addItems(cols)
        summary = self.df.describe(include='all').T
        summary['Missing'] = self.df.isna().sum()
        self.summary_text.setText(str(summary))
        self.heatmap_ax.clear()
        num_df = self.df.select_dtypes(include=[np.number])
        sns.heatmap(num_df.corr(), ax=self.heatmap_ax, cmap='coolwarm', annot=False)
        self.heatmap_canvas.draw()

    def run_training(self):
        sel = [i.text() for i in self.feature_list.selectedItems()]

        tgt = self.target_box.currentText()
        if not sel or not tgt:
            return
        df = self.df[sel + [tgt]].copy()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.apply(pd.to_numeric, errors='coerce')
        df.dropna(inplace=True)

        model_type = self.model_box.currentText()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        forecast_steps = self.forecast_spin.value()
        if model_type == 'NeuroPRINv4':
            L, R = self.seq_spin.value(), self.regime_spin.value()
            try:
                dfp = prune_data(df)
                dfr = compute_fourier_resonance(dfp, window_size=L)
                dfq = compute_rqa_metrics(dfr, window_size=L)
                X, y, chaos, _, regs, _ = prepare_sequences_with_prin_plus_plus(
                    {'_': dfq}, seq_length=L, n_features=len(sel), n_regimes=R
                )
            except Exception as e:
                self.summary_text.setText(f"Feature error: {e}")
                return
            if len(X) < 10:
                self.summary_text.setText("Too small after prep.")
                return

            split = int(len(X) * 0.8)
            Xt, Xv = X[:split], X[split:]
            yt, yv = y[:split], y[split:]
            # assuming `L` is your sequence length
            val_dates = dfq.index[L + split : L + split + len(yv)]


            ct, cv = chaos[:split], chaos[split:]
            rt, rv = regs[:split], regs[split:]

            ys = StandardScaler()
            yt_s = ys.fit_transform(yt.reshape(-1, 1))
            yv_s = ys.transform(yv.reshape(-1, 1))
            Xv_s = Xv  

            model = NeuroPRINv4(input_size=X.shape[-1], seq_len=L, num_regimes=R).to(device)
            Xt_t = torch.tensor(Xt, dtype=torch.float32).to(device)
            Xv_t = torch.tensor(Xv, dtype=torch.float32).to(device)
            yt_t = torch.tensor(yt_s, dtype=torch.float32).to(device)
            yv_t = torch.tensor(yv_s, dtype=torch.float32).to(device)
            ct_t = torch.tensor(ct, dtype=torch.float32).to(device)
            cv_t = torch.tensor(cv, dtype=torch.float32).to(device)
            rt_t = torch.tensor(rt, dtype=torch.long).to(device)
            rv_t = torch.tensor(rv, dtype=torch.long).to(device)

            opt = optim.Adam(model.parameters(), lr=self.lr_spin.value())
            loss_fn = nn.MSELoss()
            hist = {'loss': [], 'val': []}
            for _ in range(self.epochs_spin.value()):
                model.train()
                opt.zero_grad()
                out = model(Xt_t, ct_t, rt_t)
                loss = loss_fn(out, yt_t)
                loss.backward()
                opt.step()
                model.eval()
                with torch.no_grad():
                    vo = model(Xv_t, cv_t, rv_t)
                    vloss = loss_fn(vo, yv_t)
                hist['loss'].append(loss.item())
                hist['val'].append(vloss.item())

            self.plot_loss(hist)

            # === Forecast future values ===
            model.eval()
            forecast = []
            last_input = Xv_t[-1:].clone()

            with torch.no_grad():
                # ① run one extra step
                for _ in range(forecast_steps + 1):
                    pred = model(last_input, cv_t[-1:], rv_t[-1:])
                    forecast.append(pred.cpu().numpy().flatten()[0])
                    last_input = torch.cat([last_input[:, 1:], pred.unsqueeze(1)], dim=1)

            # ② drop the duplicate zero-step, keep exactly N
            forecast = np.array(forecast[:forecast_steps]).reshape(-1, 1)
            forecast = ys.inverse_transform(forecast)


            # Full predictions for validation set

            # full predictions for the validation window
            yp_val = model(Xv_t, cv_t, rv_t).cpu().numpy()
            yv_true = ys.inverse_transform(yv_t.cpu().numpy())
            yp_val   = ys.inverse_transform(yp_val)

            # compute a DatetimeIndex for our forecast horizon
            # assume uniform spacing in val_dates
            delta = val_dates[-1] - val_dates[-2]
            forecast_dates = [val_dates[-1] + delta*(i+1) for i in range(forecast_steps)]

            # hand off real dates + arrays into our new plotting helper
            self.plot_preds_from_numpy(dates=val_dates,yv=yv_true,yp=yp_val,forecast=forecast.flatten(),forecast_dates=forecast_dates)



        else:
            X = df[sel].values.astype(np.float32)
            y = df[tgt].values.astype(np.float32).reshape(-1, 1)
            split = int(len(X) * 0.8)
            dates = df.index[split:]
            Xt, Xv = X[:split], X[split:]
            yt, yv = y[:split], y[split:]
            # 1️⃣ extract the DateTimeIndex for your validation window
            val_dates = self.df.index[split : split + len(yv)]


            input_size = X.shape[1]
            model_map = {
                'BaselineLSTM': lambda: BaselineLSTM(input_size, 64, 1).to(device),
                'PRIN_LSTM': lambda: PRIN_LSTM(input_size, 64, 1).to(device),
                'DPRIN_LSTM': lambda: DPRIN_LSTM(input_size, 4, 1).to(device),
                'LinearRegression': lambda: LinearRegressionModel(),
                'RandomForest': lambda: RandomForestModel(),
                'XGBoost': lambda: XGBoostModel(),
                'SVR': lambda: SupportVectorModel(),
                'KNN': lambda: KNNModel(),
                'GradientBoosting': lambda: GradientBoostingModel()
            }
            model = model_map[model_type]()

            sX, sy = StandardScaler().fit(Xt), StandardScaler().fit(yt)
            Xt_s, Xv_s = sX.transform(Xt), sX.transform(Xv)
            yt_s, yv_s = sy.transform(yt), sy.transform(yv)



            if hasattr(model, 'fit'):
                model.fit(Xt_s, yt_s.ravel())
                yp = model.predict(Xv_s).reshape(-1, 1)
                yp = sy.inverse_transform(yp)
                yv = sy.inverse_transform(yv_s)

    # === Forecast future values for sklearn-like models ===

                forecast = []
                last_input = Xv_s[-1:].copy()

                with torch.no_grad():
                    # ① one extra iteration
                    for _ in range(forecast_steps + 1):
                        inp = torch.tensor(last_input,dtype=torch.float32).unsqueeze(1).to(device)
                        pred = model(inp)
                        val  = pred.cpu().numpy().flatten()[0]
                        forecast.append(val)
                        last_input = np.roll(last_input, -1, axis=1)
                        last_input[0, -1] = val

                # ② drop the duplicate, keep exactly forecast_steps
                forecast = np.array(forecast[:forecast_steps]).reshape(-1, 1)

                forecast = sy.inverse_transform(forecast)

    # 1️⃣ slice out exactly the val‐set timestamps
                val_dates = df.index[ split : split + len(yv) ]

                # 2️⃣ build matching forecast timestamps
                delta = val_dates[-1] - val_dates[-2]
                forecast_dates = [val_dates[-1] + delta * (i+1) for i in range(forecast_steps)]

                # 3️⃣ plot actual + predicted + forecast with real datetimes
                self.plot_preds_from_numpy(val_dates,yv,yp,forecast=forecast,forecast_dates=forecast_dates)

    # 4️⃣ (optional) extend x-axis so you can see your forecast out to last date
                self.pred_ax.set_xlim(val_dates[0], forecast_dates[-1])
                self.pred_canvas.draw()



            else:
                Xt_t = torch.tensor(Xt_s, dtype=torch.float32).unsqueeze(1).to(device)
                Xv_t = torch.tensor(Xv_s, dtype=torch.float32).unsqueeze(1).to(device)
                yt_t = torch.tensor(yt_s, dtype=torch.float32).to(device)
                yv_t = torch.tensor(yv_s, dtype=torch.float32).to(device)

                opt = optim.Adam(model.parameters(), lr=self.lr_spin.value())
                loss_fn = nn.MSELoss()
                hist = {'loss': [], 'val': []}
                for _ in range(self.epochs_spin.value()):
                    model.train()
                    opt.zero_grad()
                    out = model(Xt_t)
                    loss = loss_fn(out, yt_t)
                    loss.backward()
                    opt.step()
                    model.eval()
                    with torch.no_grad():
                        vloss = loss_fn(model(Xv_t), yv_t)
                    hist['loss'].append(loss.item())
                    hist['val'].append(vloss.item())
                self.plot_loss(hist)

# Forecast for sklearn-based models (assumes same features persist)
                                
                forecast = []

                if hasattr(model, 'predict'):  # sklearn models
                    last_input = Xv_s[-1:].copy()
                    for _ in range(forecast_steps):
                        next_pred = model.predict(last_input).reshape(1, -1)
                        forecast.append(next_pred[0, 0])
                        last_input = np.roll(last_input, -1, axis=1)
                        last_input[0, -1] = next_pred[0, 0]

                else:  # torch models
                    last_input = Xv_s[-1:].copy()
                    with torch.no_grad():
                        for _ in range(forecast_steps):
                            input_tensor = torch.tensor(last_input, dtype=torch.float32).unsqueeze(1).to(device)
                            next_pred = model(input_tensor)
                            val = next_pred.cpu().numpy().flatten()[0]
                            forecast.append(val)
                            last_input = np.roll(last_input, -1, axis=1)
                            last_input[0, -1] = val


   

                forecast = np.array(forecast[:forecast_steps]).reshape(-1, 1)

                forecast = sy.inverse_transform(forecast)
                with torch.no_grad():
                    yp_tensor = model(Xv_t)
                yp = yp_tensor.cpu().numpy()
                yp = sy.inverse_transform(yp)
                yv = sy.inverse_transform(yv_s)


                # 1️⃣ slice out exactly the val-set timestamps from YOUR cleaned df
                val_dates = df.index[ split : split + len(yv) ]

                # 2️⃣ try to infer the frequency; fall back to a manual delta if needed
                freq = val_dates.freq or pd.infer_freq(val_dates)
                if freq:
                # +1 so we don’t include the last known point
                    forecast_dates = pd.date_range(start=val_dates[-1],periods=forecast_steps + 1,freq=freq)[1:]

                else:

                    delta = val_dates[-1] - val_dates[-2]
                    forecast_dates = [val_dates[-1] + delta * (i+1)
                    for i in range(forecast_steps)
                    ]

                # 3️⃣ now plot history & forecast with real datetimes
                self.plot_preds_from_numpy(dates=val_dates,yv=yv,yp=yp,forecast=forecast,forecast_dates=forecast_dates)


                # 4️⃣ extend the x-axis so you actually see your 5 forecast points
                self.pred_ax.set_xlim(val_dates[0], forecast_dates[-1])
                self.pred_canvas.draw()

                # (and then continue calling your torch plot if you still want it)
                self.plot_preds(model, Xv_t, yv_t, scaler_y=sy)


            return

    def plot_loss(self, hist):
        self.loss_ax.clear()
        self.loss_ax.plot(hist['loss'], label='Train')
        self.loss_ax.plot(hist['val'], label='Val')
        self.loss_ax.legend()
        self.loss_ax.set_title('Loss')
        self.loss_fig.tight_layout()
        self.loss_canvas.draw()

    def plot_preds(self, model, Xv_t, yv_t, scaler_y, chaos=None, regime=None):
        model.eval()
        with torch.no_grad():
            if chaos is not None:
                yp = model(Xv_t, chaos, regime).cpu().numpy()
            else:
                yp = model(Xv_t).cpu().numpy()
        yp = scaler_y.inverse_transform(yp)
        yv = scaler_y.inverse_transform(yv_t.cpu().numpy())




    def plot_preds_from_numpy(self,dates: pd.DatetimeIndex,yv: np.ndarray,yp: np.ndarray,forecast: Optional[np.ndarray] = None,forecast_dates:Optional[List[pd.Timestamp]] = None):
        self.pred_ax.clear()
        # plot actual vs predicted on the real dates
        self.pred_ax.plot(dates, yv, label='Actual')
        self.pred_ax.plot(dates, yp, label='Predicted')



        if forecast is not None and forecast_dates is not None:
            self.pred_ax.plot(forecast_dates, forecast,label='Forecast',linestyle='--')
            # draw a vertical line at the boundary between history & forecast
            self.pred_ax.axvline(x=dates[-1], linestyle=':', color='gray', label='Forecast start')

        # format x-axis as dates
        self.pred_ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        self.pred_fig.autofmt_xdate()


        mae = mean_absolute_error(yv, yp)
        rmse = mean_squared_error(yv, yp, squared=False)
        r2 = r2_score(yv, yp)
        self.pred_ax.set_title(f'MAE={mae:.3f} RMSE={rmse:.3f} R²={r2:.3f}')
        self.pred_ax.legend()
        self.pred_fig.tight_layout()
        self.pred_canvas.draw()

        self.pred_table.clear()
        forecast_len = len(forecast) if forecast is not None else 0
        total_rows = len(yv) + forecast_len

        self.pred_table.setRowCount(total_rows)
        self.pred_table.setColumnCount(3)
        self.pred_table.setHorizontalHeaderLabels(["Idx", "Actual", "Pred"])

        # Fill known predictions
        for i, (av, pv) in enumerate(zip(yv, yp)):
            self.pred_table.setItem(i, 0, QTableWidgetItem(str(i)))
            self.pred_table.setItem(i, 1, QTableWidgetItem(f"{av[0]:.3f}"))
            self.pred_table.setItem(i, 2, QTableWidgetItem(f"{pv[0]:.3f}"))

        # Append forecasted values
        if forecast is not None:
            for j, fv in enumerate(forecast.flatten()):
                idx = len(yv) + j
                self.pred_table.setItem(idx, 0, QTableWidgetItem(str(idx)))
                self.pred_table.setItem(idx, 1, QTableWidgetItem("Future"))
                self.pred_table.setItem(idx, 2, QTableWidgetItem(f"{fv:.3f}"))

        self.pred_table.resizeColumnsToContents()
        # at the end of your forecast & plot code:
        self.tabs.setCurrentIndex(3)   # zero-based index of the “Predictions” tab



if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = NeuroPRINGUI()
    win.show()
    sys.exit(app.exec_())
