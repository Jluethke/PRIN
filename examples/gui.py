#!/usr/bin/env python3
"""
examples/gui.py

Graphical interface for NeuroPRIN demos with dynamic column overlays:
 - Toggle between 'keras' and 'torch' pipelines
 - Enter symbols, date range, seq_len, epochs, batch_size
 - Automatically loads local CSVs from /stockdata/[SYMBOL]/[SYMBOL].csv if present,
   otherwise falls back to API fetch via neuroprin.data.load_price_data
 - Computes indicators, displays raw data table
 - Dynamically select any columns (raw or generated) to overlay on the price chart
 - Lists output files and allows opening them
"""

import os
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from neuroprin.train import run_keras_models, run_pytorch_pipeline
from neuroprin.data import load_price_data, compute_indicators


class NeuroPRINGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("NeuroPRIN Demo GUI")
        self.geometry("1200x800")
        self._build_widgets()
        self.df = None

    def _build_widgets(self):
        # Control panel
        ctrl = ttk.Frame(self)
        ctrl.grid(row=0, column=0, sticky="nw", padx=10, pady=10)

        # Pipeline selector
        ttk.Label(ctrl, text="Pipeline:").grid(row=0, column=0, sticky="w")
        self.pipeline_var = tk.StringVar(value="torch")
        ttk.Combobox(ctrl, textvariable=self.pipeline_var,
                     values=["keras", "torch"], state="readonly")\
            .grid(row=0, column=1, sticky="ew")

        # Symbols entry
        ttk.Label(ctrl, text="Symbols (comma-separated):")\
            .grid(row=1, column=0, sticky="w")
        self.symbols_entry = ttk.Entry(ctrl)
        self.symbols_entry.grid(row=1, column=1, sticky="ew")

        # Date range
        ttk.Label(ctrl, text="Start Date (YYYY-MM-DD):")\
            .grid(row=2, column=0, sticky="w")
        self.start_entry = ttk.Entry(ctrl)
        self.start_entry.grid(row=2, column=1, sticky="ew")
        ttk.Label(ctrl, text="End Date (YYYY-MM-DD):")\
            .grid(row=3, column=0, sticky="w")
        self.end_entry = ttk.Entry(ctrl)
        self.end_entry.grid(row=3, column=1, sticky="ew")

        # Sequence length
        ttk.Label(ctrl, text="Seq Length:").grid(row=4, column=0, sticky="w")
        self.seq_len_spin = ttk.Spinbox(ctrl, from_=1, to=200, increment=1)
        self.seq_len_spin.set("20")
        self.seq_len_spin.grid(row=4, column=1, sticky="ew")

        # Epochs
        ttk.Label(ctrl, text="Epochs:").grid(row=5, column=0, sticky="w")
        self.epochs_spin = ttk.Spinbox(ctrl, from_=1, to=1000, increment=1)
        self.epochs_spin.set("50")
        self.epochs_spin.grid(row=5, column=1, sticky="ew")

        # Batch size
        ttk.Label(ctrl, text="Batch Size:").grid(row=6, column=0, sticky="w")
        self.batch_spin = ttk.Spinbox(ctrl, from_=1, to=1024, increment=1)
        self.batch_spin.set("64")
        self.batch_spin.grid(row=6, column=1, sticky="ew")

        # Run button
        self.run_button = ttk.Button(ctrl, text="Run Pipeline",
                                     command=self._on_run)
        self.run_button.grid(row=7, column=0, columnspan=2, pady=10)

        # Column selector for overlays
        ttk.Label(ctrl, text="Select columns to overlay:")\
            .grid(row=8, column=0, columnspan=2, sticky="w", pady=(10,0))
        self.columns_listbox = tk.Listbox(ctrl, selectmode="multiple", height=10)
        self.columns_listbox.grid(row=9, column=0, columnspan=2, sticky="nsew")
        ctrl.rowconfigure(9, weight=1)

        # Notebook for raw data and chart
        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=0, column=1, rowspan=12,
                           sticky="nsew", padx=10, pady=10)

        # Raw data tab
        self.raw_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.raw_frame, text="Raw Data")
        self.tree = ttk.Treeview(self.raw_frame)
        self.tree.pack(expand=True, fill="both")

        # Chart tab
        self.chart_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.chart_frame, text="Chart")
        self.fig, self.ax = plt.subplots(figsize=(6,4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas.get_tk_widget().pack(expand=True, fill="both")

        # Output files list
        ttk.Label(self, text="Output Files:")\
            .grid(row=12, column=0, sticky="w", padx=10)
        self.files_list = tk.Listbox(self)
        self.files_list.grid(row=13, column=0, columnspan=2,
                             sticky="ew", padx=10, pady=(0,10))
        ttk.Button(self, text="Open File", command=self._on_open_file)\
            .grid(row=13, column=2, sticky="e", padx=10, pady=(0,10))

        # Configure resizing behavior
        self.columnconfigure(1, weight=1)
        self.rowconfigure(13, weight=1)

    def _on_run(self):
        # Gather inputs
        symbols = [s.strip() for s in self.symbols_entry.get().split(",") if s.strip()]
        start = self.start_entry.get().strip()
        end = self.end_entry.get().strip()
        try:
            seq_len = int(self.seq_len_spin.get())
            epochs = int(self.epochs_spin.get())
            batch_size = int(self.batch_spin.get())
        except ValueError:
            messagebox.showerror(
                "Input Error",
                "Sequence length, epochs, and batch size must be integers."
            )
            return

        pipeline = self.pipeline_var.get()
        output_dir = f"{pipeline}_gui_runs"
        os.makedirs(output_dir, exist_ok=True)

        # Run selected pipeline
        try:
            if pipeline == "keras":
                run_keras_models(
                    symbols=symbols,
                    start_date=start,
                    end_date=end,
                    seq_len=seq_len,
                    test_size=0.2,
                    epochs=epochs,
                    batch_size=batch_size,
                    output_dir=output_dir,
                )
            else:
                run_pytorch_pipeline(
                    symbols=symbols,
                    start_date=start,
                    end_date=end,
                    seq_len=seq_len,
                    test_size=0.2,
                    batch_size=batch_size,
                    epochs=epochs,
                    device="cpu",
                    output_dir=output_dir,
                )
        except Exception as e:
            messagebox.showerror("Pipeline Error", str(e))
            return

        # Load and prepare data
        self.df = self._load_local_or_api(symbols, start, end)
        self.df = compute_indicators(self.df).reset_index()

        # Update UI elements
        self._show_raw_data(self.df)
        self._populate_columns(self.df)
        self._plot_selected_columns()

        # List output files
        self._list_output_files(output_dir)

    def _load_local_or_api(self, symbols, start, end):
        frames = []
        for sym in symbols:
            local_file = os.path.join("stockdata", sym, f"{sym}.csv")
            if os.path.exists(local_file):
                df = pd.read_csv(local_file, parse_dates=["Date"])
            else:
                df = load_price_data([sym], start, end)
            df["Symbol"] = sym
            frames.append(df)
        return pd.concat(frames, ignore_index=True)

    def _show_raw_data(self, df: pd.DataFrame):
        self.tree.delete(*self.tree.get_children())
        self.tree["columns"] = list(df.columns)
        self.tree["show"] = "headings"
        for col in df.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100, anchor="center")
        for _, row in df.iterrows():
            self.tree.insert("", "end", values=list(row))

    def _populate_columns(self, df: pd.DataFrame):
        self.columns_listbox.delete(0, tk.END)
        for col in df.columns:
            self.columns_listbox.insert(tk.END, col)

    def _plot_selected_columns(self):
        if self.df is None:
            return
        self.ax.clear()
        selected_indices = self.columns_listbox.curselection()
        for idx in selected_indices:
            col = self.columns_listbox.get(idx)
            if col in ("Date", "Symbol"):
                continue
            for sym in self.df["Symbol"].unique():
                sub = self.df[self.df["Symbol"] == sym]
                self.ax.plot(sub["Date"], sub[col], label=f"{sym} {col}")
        self.ax.set_title("Selected Data Overlays")
        self.ax.legend(loc="upper left", fontsize="small")
        self.canvas.draw()

    def _list_output_files(self, output_dir):
        self.files_list.delete(0, tk.END)
        for fname in sorted(os.listdir(output_dir)):
            path = os.path.join(output_dir, fname)
            self.files_list.insert(tk.END, path)

    def _on_open_file(self):
        sel = self.files_list.curselection()
        if not sel or not hasattr(self, "files_list"):
            return
        path = self.files_list.get(sel[0])
        if os.name == "nt":
            os.startfile(path)
        else:
            os.system(f'xdg-open "{path}"')

if __name__ == "__main__":
    app = NeuroPRINGUI()
    app.mainloop()
