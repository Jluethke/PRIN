# \# PRIN

# 

# 📘 Predictive Resonance Inference Networks (PRIN) — README

# Overview

# 

# Predictive Resonance Inference Networks (PRIN) is a theorem-driven neuromorphic architecture designed for robust forecasting of nonlinear, chaotic financial time series. This repository contains:

# 

# PRIN \& NeuroPRINv4 model architectures

# 

# Chaos gating, resonance amplification, predictive coding contraction, and regime gating

# 

# Full PRIN++ preprocessing (RQA, HMM regimes, chaos/resonance sequences)

# 

# NeuroPRIN Studio Pro v5 GUI

# 

# Comprehensive empirical validation suite

# 

# Baseline models for benchmarking (ARIMA, Transformer, TCN, classical ML)

# 

# 🔬 Scientific Motivation

# 

# Financial markets exhibit:

# 

# Chaos (sensitivity to initial conditions)

# 

# Regime shifts

# 

# Nonlinear multi-timescale behavior

# 

# Spectral resonance and memory effects

# 

# Traditional ML models struggle with this structure.

# PRIN introduces four foundational theorems that guarantee stability and meaningful learning:

# 

# Theorem 1 — Chaos Gating Contraction

# 

# The chaos gate σ(αλ̂ₜ) ensures features contract toward a stable manifold.

# 

# Theorem 2 — Pruning Index Stability

# 

# Salience masks constrain the hidden-state norm to bounded trajectories.

# 

# Theorem 3 — Predictive Coding Contraction

# 

# The operator

# 

# 𝑀

# =

# 𝐼

# −

# 𝜂

# (

# 𝐼

# −

# 𝑊

# )

# M=I−η(I−W)

# 

# guarantees contraction and stable error propagation.

# 

# Theorem 4 — Resonance Dominance

# 

# Resonant patterns rₜ produce amplifying gates that prioritize meaningful periodic structure.

# 

# Theorem 5 — Regime Coherence and Transition Stability (PRIN++)

# 

# HMM segmentation ensures regime-aware feature allocation.

# 

# All five theorems are empirically validated through the automated testing suite.

# 

# 🧠 Model Architectures

# Baseline Models

# 

# Linear Regression

# 

# Random Forest

# 

# XGBoost

# 

# SVR

# 

# KNN

# 

# Gradient Boosting

# 

# Baseline LSTMs

# 

# PRIN Architectures

# 

# PRIN-LSTM

# 

# DPRIN-LSTM — dynamic gating

# 

# NeuroPRINv4 — full PRIN++

# 

# Chaos gating

# 

# Resonance amplification

# 

# Predictive coding

# 

# Regime embedding

# 

# RQA inputs

# 

# Multi-task outputs (return, direction, volatility)

# 

# 📊 Empirical Validation Suite

# 

# The file Empirical\_Output.py runs a complete, automated, publication-ready analysis suite including:

# 

# Chaos Analysis

# 

# chaos\_map.pdf

# 

# chaos\_histogram.pdf

# 

# chaos\_gate.pdf

# 

# chaos\_contraction\_effect.pdf

# 

# chaos\_variance\_reduction.pdf

# 

# Resonance Analysis

# 

# resonance\_strength.pdf

# 

# resonance\_gate.pdf

# 

# spectral\_heatmap.pdf

# 

# resonance\_crosscorr.pdf

# 

# Regime Analysis

# 

# regime\_timeline.pdf

# 

# regime\_transition\_matrix.pdf

# 

# rolling\_dm\_regimes.tex

# 

# Predictive Coding (Theorem 3)

# 

# predictive\_coding\_contraction.pdf

# 

# predictive\_coding\_error\_trajectory.pdf

# 

# spectral\_radius.pdf

# 

# jacobian\_norm.pdf

# 

# Salience \& Pruning

# 

# pruning\_mask.pdf

# 

# pruning\_effect.pdf

# 

# pruning\_sparsity.pdf

# 

# Forecast Diagnostics

# 

# forecast\_vs\_actual.pdf

# 

# error\_distribution.pdf

# 

# Statistical Tests

# 

# Rolling DM test

# 

# Paired t-test

# 

# Model Confidence Set (MCS)

# 

# Chaotic bootstrap test

# 

# LaTeX tables:

# 

# stat\_tests.tex

# 

# mcs\_results.tex

# 

# LaTeX Appendix for Reviewers

# 

# empirical\_appendix.tex

# 

# This suite provides all empirical evidence required for a PRIN publication.

# 

# 🖥️ NeuroPRIN Studio Pro v5 GUI

# 

# The GUI (gui.py) provides:

# 

# CSV loading

# 

# Feature/target selection

# 

# Model selection

# 

# Hyperparameter inputs

# 

# Automated training

# 

# Forecast visualization

# 

# Metrics display

# 

# Correlation heatmaps

# 

# Full PRIN empirical diagnostics with progress bar

# 

# This is your research studio for PRIN development.

# 

# 🚀 Installation

# Clone repo

# git clone https://github.com/jluethke/PRIN.git

# cd PRIN

# 

# Install dependencies

# pip install -r requirements.txt
# pip install PandasTA-v0.3.14b source code.tar.gz
# 

# 

# Make sure PyQt5, PyTorch, NumPy, Pandas, Seaborn, and Statsmodels are included.

# 

# ▶️ Running the GUI

# python gui.py

# 

# 

# This launches NeuroPRIN Studio Pro v5.

# 

# 📚 Running the Empirical Suite Manually

# 

# If you want to evaluate an already-trained model:

# 

# from Empirical\_Output import run\_all\_empirical\_tests

# 

# run\_all\_empirical\_tests(

# &nbsp;   model,

# &nbsp;   Xv,

# &nbsp;   chaos,

# &nbsp;   resonance,

# &nbsp;   regimes,

# &nbsp;   rqa,

# &nbsp;   y\_true,

# &nbsp;   y\_pred

# )

# 

# 

# Results go to:

# 

# empirical/figures/

# empirical/tables/

# empirical/latex/

# 

# 📈 Training from Code

# 

# Example:

# 

# from train import train\_model

# from models import NeuroPRINv4

# 

# model = NeuroPRINv4(input\_size=10, seq\_len=30, num\_regimes=3)

# train\_model(model, X\_train, y\_train)

# 

# 📄 Citing PRIN

# 

# Use this if someone wants to cite PRIN:

# 

# Luethke (2025). Predictive Resonance Inference Networks (PRIN): A Theorem-Driven Framework for Forecasting Chaotic Financial Time Series.

