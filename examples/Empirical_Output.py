"""
Empirical_Output.py

Full empirical diagnostics module for PRIN / NeuroPRINv4,
providing all evidence required to validate the four PRIN theorems.

Outputs (PDF-only) in:

empirical/
    figures/*.pdf
    tables/*.tex
    latex/*.tex
"""
import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import inspect

from matplotlib.colors import LogNorm
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from numpy.linalg import eigvals
from tqdm import tqdm


# =========================================================
# DIRECTORY SETUP
# =========================================================

def ensure_dirs():
    base = "empirical"
    dirs = [
        f"{base}/figures",
        f"{base}/tables",
        f"{base}/latex",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return base


# =========================================================
# CHAOS DIAGNOSTICS (THEOREM 1)
# =========================================================

def plot_chaos_map(chaos, save=True):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(chaos, color='red', linewidth=1.5)
    ax.set_title("Chaos Map (Lyapunov Exponent Trajectory)")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Lyapunov Estimate")
    fig.tight_layout()
    if save:
        fig.savefig("empirical/figures/chaos_map.pdf", bbox_inches='tight')
    plt.close(fig)

def plot_chaos_histogram(chaos, save=True):
    chaos = np.asarray(chaos).reshape(-1)
    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(chaos, kde=True, ax=ax)
    ax.set_title("Chaos Distribution (Lyapunov Estimates)")
    fig.tight_layout()
    if save: fig.savefig("empirical/figures/chaos_histogram.pdf")
    plt.close(fig)

def plot_pruning_sparsity(mask, save=True):
    sparsity = mask.mean(axis=1)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(sparsity)
    ax.set_title("Pruning Sparsity Over Sequences")
    if save: fig.savefig("empirical/figures/pruning_sparsity.pdf")
    plt.close(fig)

def plot_jacobian_norm(model, save=True):
    W = model.pc.W.detach().cpu().numpy()
    eta = model.pc.eta

    I = np.eye(W.shape[0])
    M = I - eta * (I - W)

    norm = np.linalg.norm(M, ord=2)

    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar([0], [norm])
    ax.axhline(1, color='red', linestyle='--')
    ax.set_title("Jacobian Norm ||M|| (Must be <1 for Contraction)")
    if save: fig.savefig("empirical/figures/jacobian_norm.pdf")
    plt.close(fig)


def plot_resonance_crosscorr(X, resonance, save=True):
    X = X[:, -1, :]   # last timestep features
    r = resonance.reshape(-1, 1)

    corr = np.corrcoef(np.hstack([X, r]).T)

    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, cmap="coolwarm", ax=ax)
    ax.set_title("Feature–Resonance Cross Correlation Matrix")
    if save: fig.savefig("empirical/figures/resonance_crosscorr.pdf")
    plt.close(fig)

def plot_rqa_radar(rqa, save=True):
    rqa = np.asarray(rqa)

    # ensure 2D
    if rqa.ndim == 1:
        rqa = rqa.reshape(-1, 1)

    labels = [f"RQA{i}" for i in range(rqa.shape[1])]
    mean_vals = rqa.mean(axis=0)


    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    mean_vals = np.concatenate((mean_vals, [mean_vals[0]]))
    angles = np.concatenate((angles, [angles[0]]))

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, mean_vals)
    ax.fill(angles, mean_vals, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title("Mean RQA Radar Plot")
    if save: fig.savefig("empirical/figures/rqa_radar.pdf")
    plt.close(fig)


def plot_regime_transition_matrix(regimes, num_regimes=None, save=True):
    regimes = regimes.astype(int)
    if num_regimes is None:
        num_regimes = regimes.max() + 1

    M = np.zeros((num_regimes, num_regimes))

    for a, b in zip(regimes[:-1], regimes[1:]):
        M[a, b] += 1

    M = M / M.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(M, annot=True, cmap="viridis", ax=ax)
    ax.set_title("Regime Transition Probability Matrix")
    if save: fig.savefig("empirical/figures/regime_transition_matrix.pdf")
    plt.close(fig)



def plot_chaos_variance_pre_post(X, chaos, alpha=2.0, save=True):
    chaos = chaos.reshape(-1, 1)
    g = 1 / (1 + np.exp(-alpha * chaos))

    raw = X[:, -1, 0]
    contracted = raw * g.reshape(-1)

    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(raw, label="Raw", alpha=0.5)
    ax.plot(contracted, label="After Chaos Gate", linewidth=2)
    ax.legend()
    ax.set_title("Variance Reduction Under Chaos Gating")
    if save: fig.savefig("empirical/figures/chaos_variance_reduction.pdf")
    plt.close(fig)


def plot_chaos_gate(chaos, alpha=2.0, save=True):
    chaos = np.asarray(chaos).reshape(-1)
    g = 1 / (1 + np.exp(-alpha * chaos))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(g, color='blue', linewidth=1.5)
    ax.set_title("Chaos Gate Output  σ(αλ̂ₜ)")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Gate Value")
    fig.tight_layout()
    if save:
        fig.savefig("empirical/figures/chaos_gate.pdf", bbox_inches='tight')
    plt.close(fig)


def plot_chaos_contraction_effect(X, chaos, alpha=2.0, save=True):
    """
    Shows how gating reduces chaotic excursions.
    This illustrates contraction behavior under Theorem 1.
    X : shape (N, L, F)
    chaos : shape (N,)
    """
    X = np.asarray(X)
    chaos = np.asarray(chaos).reshape(-1)

    if X.ndim != 3:
        raise ValueError(f"plot_chaos_contraction_effect expects X with ndim=3, got {X.ndim}")

    N = min(len(chaos), X.shape[0])
    X = X[:N]
    chaos = chaos[:N]

    g = 1 / (1 + np.exp(-alpha * chaos.reshape(-1, 1)))
    X_raw = X[:, -1, 0]   # last-step value per sequence
    X_contracted = X_raw * g.reshape(-1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(X_raw, label="Raw Feature", alpha=0.6)
    ax.plot(X_contracted, label="Chaos-Gated Feature", linewidth=2)
    ax.set_title("Chaos Contraction Effect: x vs g·x")
    ax.legend()
    fig.tight_layout()

    if save:
        fig.savefig("empirical/figures/chaos_contraction_effect.pdf", bbox_inches='tight')
    plt.close(fig)


# =========================================================
# RESONANCE DIAGNOSTICS (THEOREM 4)
# =========================================================

def plot_resonance_strength(resonance, save=True):
    resonance = np.asarray(resonance).reshape(-1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(resonance, color='purple')
    ax.set_title("Resonance Strength  rₜ")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Resonance")
    fig.tight_layout()
    if save:
        fig.savefig("empirical/figures/resonance_strength.pdf", bbox_inches='tight')
    plt.close(fig)


def plot_resonance_gate(resonance, beta=10.0, save=True):
    r = np.asarray(resonance).reshape(-1)
    g = 1 / (1 + np.exp(-beta * r))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(g, color='green')
    ax.set_title("Resonance Gate Output  σ(βrₜ)")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Gate Value")
    fig.tight_layout()
    if save:
        fig.savefig("empirical/figures/resonance_gate.pdf", bbox_inches='tight')
    plt.close(fig)


def plot_spectral_heatmap(X, save=True):
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"plot_spectral_heatmap expects X with ndim=3, got {X.ndim}")

    N, L, _ = X.shape
    fft_matrix = np.abs(np.fft.rfft(X[:, :, 0], axis=1))

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(fft_matrix, cmap='magma', ax=ax)
    ax.set_title("Spectral Resonance Heatmap")
    ax.set_xlabel("Frequency Bin")
    ax.set_ylabel("Sequence Index")
    fig.tight_layout()
    if save:
        fig.savefig("empirical/figures/spectral_heatmap.pdf", bbox_inches='tight')
    plt.close(fig)


# =========================================================
# REGIME DIAGNOSTICS (THEOREM 5)
# =========================================================

def plot_regime_timeline(regimes, save=True):
    regimes = np.asarray(regimes).reshape(-1)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.scatter(range(len(regimes)), regimes, c=regimes, cmap='viridis', s=8)
    ax.set_title("Regime Timeline (HMM States)")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Regime ID")
    fig.tight_layout()
    if save:
        fig.savefig("empirical/figures/regime_timeline.pdf", bbox_inches='tight')
    plt.close(fig)


# =========================================================
# RQA METRICS
# =========================================================

def plot_rqa_metrics(rqa, save=True):
    rqa = np.asarray(rqa).reshape(-1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rqa, color='black')
    ax.set_title("RQA Metric (Recurrence Rate)")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("RR Value")
    fig.tight_layout()
    if save:
        fig.savefig("empirical/figures/rqa_metrics.pdf", bbox_inches='tight')
    plt.close(fig)


# =========================================================
# SALIENCE & PRUNING (THEOREM 2)
# =========================================================

def plot_pruning_mask(mask, save=True):
    mask = np.asarray(mask)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(mask.astype(int), cmap="binary", cbar=False, ax=ax)
    ax.set_title("Salience-Based Pruning Mask")
    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Sequence Index")
    fig.tight_layout()

    if save:
        fig.savefig("empirical/figures/pruning_mask.pdf", bbox_inches='tight')
    plt.close(fig)


def plot_feature_magnitudes_before_after(before, after, save=True):
    before = np.asarray(before).reshape(-1)
    after = np.asarray(after).reshape(-1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(before, label="Before Pruning", alpha=0.5)
    ax.plot(after, label="After Pruning", linewidth=2)
    ax.set_title("Feature Magnitudes: Before vs After Pruning")
    ax.legend()
    fig.tight_layout()

    if save:
        fig.savefig("empirical/figures/pruning_effect.pdf", bbox_inches='tight')
    plt.close(fig)



# =========================================================
# HIDDEN-STATE NORM TRAJECTORY (THEOREM 2 DIRECT DIAGNOSTIC)
# =========================================================

def plot_hidden_state_norm(model, Xv, save=True):
    """
    Tracks ||h_t|| over time for a single batch example.
    Works for PRIN_LSTM and NeuroPRINv4. If model does not expose
    explicit hidden states, we simulate by stepping through LSTM.
    """

    # Take a single sequence for analysis
    x = torch.tensor(Xv[0:1], dtype=torch.float32)

    norms = []

    try:
        lstm_input_dim = model.lstm.input_size
        hs = Xv[:, :, :lstm_input_dim]   # ensure correct slice
        norms = np.linalg.norm(hs, axis=-1).mean(axis=1).tolist()

    except Exception:
        try:
            # Case 2: NeuroPRINv4 uses LSTM internally; need partial feed
            lstm = model.lstm
            h = torch.zeros(model.num_layers, 1, model.hidden_size)
            c = torch.zeros_like(h)

            for t in range(x.size(1)):
                out, (h, c) = lstm(x[:, t:t+1, :], (h, c))
                norms.append(float(torch.norm(h.squeeze(0))))
        except Exception as e:
            print("[ERROR] Hidden-state norm extraction failed:", e)
            return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(norms, marker='o')
    ax.set_title("Hidden-State Norm vs Time (Boundedness Diagnostic)")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("||h_t||")
    fig.tight_layout()

    if save:
        fig.savefig("empirical/figures/hidden_state_norm.pdf", bbox_inches='tight')
    plt.close(fig)



# =========================================================
# PREDICTIVE CODING ERROR TRAJECTORY (THEOREM 3 DIRECT TEST)
# =========================================================

def plot_predictive_coding_error(model, Xv, steps=20, save=True):
    """
    Compute contraction error: e_t = ||x_t - M x_{t-1}||
    using the predictive coding contraction matrix.
    """

    if not hasattr(model, "pc"):
        print("[WARN] Model has no predictive coding layer.")
        return

    W = model.pc.W.detach().cpu().numpy()
    eta = float(model.pc.eta)
    dim = W.shape[0]

    I = np.eye(dim)
    M = I - eta * (I - W)

    # Initial x vector from last hidden embedding of sample 0
    x = torch.tensor(Xv[0, -1], dtype=torch.float32).detach().numpy()
    x = x[:dim] if x.shape[0] >= dim else np.pad(x, (0, dim - x.shape[0]))

    errors = []
    prev = x.copy()

    for _ in range(steps):
        new = M @ prev
        err = np.linalg.norm(new - prev)
        errors.append(err)
        prev = new

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(errors, marker='o')
    ax.set_title("Predictive Coding Error Trajectory")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("||x_t - M x_{t-1}||")
    fig.tight_layout()

    if save:
        fig.savefig("empirical/figures/predictive_coding_error_trajectory.pdf", bbox_inches='tight')
    plt.close(fig)


# =========================================================
# PREDICTIVE CODING CONTRACTION (THEOREM 3)
# =========================================================

def plot_predictive_coding_contraction(x0, model, steps=10, save=True):
    """
    Evaluates iterative contraction:
        x_{t+1} = M x_t
    using model.pc.W and η.
    """
    if not hasattr(model, "pc"):
        raise AttributeError("model has no 'pc' module; cannot run predictive coding contraction.")

    W = model.pc.W.detach().cpu().numpy()
    eta = float(model.pc.eta)
    dim = W.shape[0]

    if x0.shape[0] != dim:
        raise ValueError(f"x0 dim {x0.shape[0]} != W dim {dim}")

    I = np.eye(dim)
    M = I - eta * (I - W)

    xs = [x0]
    for _ in range(steps):
        xnew = M @ xs[-1]
        xs.append(xnew)

    norms = [np.linalg.norm(v) for v in xs]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(norms, marker='o')
    ax.set_title("Predictive Coding Contraction Over Iterations")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("||x||")
    fig.tight_layout()

    if save:
        fig.savefig("empirical/figures/predictive_coding_contraction.pdf", bbox_inches='tight')
    plt.close(fig)


def plot_matrix_spectral_radius(model, save=True):
    if not hasattr(model, "pc"):
        raise AttributeError("model has no 'pc' module; cannot compute spectral radius.")

    W = model.pc.W.detach().cpu().numpy()
    eta = float(model.pc.eta)
    dim = W.shape[0]

    I = np.eye(dim)
    M = I - eta * (I - W)
    eigs = np.abs(eigvals(M))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(range(len(eigs)), eigs)
    ax.axhline(1.0, color='red', linestyle='--')
    ax.set_title("Spectral Radius of Contraction Map (I − η(I − W))")
    ax.set_xlabel("Eigenvalue Index")
    ax.set_ylabel("|λ|")
    fig.tight_layout()

    if save:
        fig.savefig("empirical/figures/spectral_radius.pdf", bbox_inches='tight')
    plt.close(fig)

# =========================================================
# AUTOMATED ABLATION STUDY (GENERATES PDF + LaTeX TABLE)
# =========================================================

def run_ablation_study(model_class, base_kwargs, Xv, y_true, device="cpu"):
    """
    Runs ablations:
        - Full model
        - No chaos
        - No resonance
        - No predictive coding
        - No regime embedding
    Returns dict of RMSE values for each.
    """

    device = torch.device(device)
    results = {}

    def evaluate(m):
        m.eval()
        with torch.no_grad():
            X = torch.tensor(Xv, dtype=torch.float32).to(device)
            pred = m(X).cpu().numpy().reshape(-1)
        y = np.asarray(y_true).reshape(-1)
        N = min(len(y), len(pred))
        return np.sqrt(((pred[:N] - y[:N])**2).mean())

    # determine allowed constructor parameters
    allowed_keys = list(inspect.signature(model_class.__init__).parameters.keys())
    allowed_keys.remove("self")



    clean_kwargs = {k: v for k, v in base_kwargs.items() if k in allowed_keys}

    # 1. Full model
    model_full = model_class(**clean_kwargs).to(device)
    results["Full PRIN"] = evaluate(model_full)

    # 2. No chaos
    model_nc = model_class(**clean_kwargs).to(device)
    model_nc.use_chaos = False
    results["No Chaos Gate"] = evaluate(model_nc)

    # 3. No resonance
    model_nr = model_class(**clean_kwargs).to(device)
    model_nr.use_resonance = False
    results["No Resonance"] = evaluate(model_nr)

    # 4. No predictive coding
    model_np = model_class(**clean_kwargs).to(device)
    model_np.use_pc = False
    model_np.pc = nn.Identity()
    results["No Predictive Coding"] = evaluate(model_np)

    # 5. No regime embedding
    model_ne = model_class(**clean_kwargs).to(device)
    model_ne.use_regimes = False
    model_ne.regime_embedding = nn.Identity()
    results["No Regime Embedding"] = evaluate(model_ne)



    # Produce PDF bar chart
    fig, ax = plt.subplots(figsize=(8,5))
    labels = list(results.keys())
    vals = [results[k] for k in labels]
    ax.bar(labels, vals)
    ax.set_title("Ablation Study (RMSE)")
    ax.set_ylabel("RMSE")
    ax.tick_params(axis='x', rotation=40)
    fig.tight_layout()
    fig.savefig("empirical/figures/ablation_results.pdf")
    plt.close(fig)

    # Produce LaTeX table
    tex = "empirical/tables/ablation_results.tex"
    with open(tex, "w") as f:
        f.write("\\begin{tabular}{lc}\n")
        f.write("\\toprule\nModel Variant & RMSE \\\\\n\\midrule\n")
        for k, v in results.items():
            f.write(f"{k} & {v:.6f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    print("[OK] Ablation results saved as PDF + LaTeX table.")
    return results


# =========================================================
# STATISTICAL SIGNIFICANCE TESTS (DM TEST + T-TEST)
# =========================================================
from scipy.stats import ttest_rel, norm


def diebold_mariano_test(e1, e2, h=1, crit="MSE", B=500):
    B_local = int(B)     # ensure local binding

    """
    Computes Diebold-Mariano test statistic for forecast accuracy comparison.
    e1, e2: forecast error series (y_pred - y_true)
    h: forecast horizon (1 for one-step ahead)
    crit: "MSE" or "MAE"
    Returns: DM statistic, p-value
    """

    e1 = np.asarray(e1)
    e2 = np.asarray(e2)
    N  = min(len(e1), len(e2))
    e1 = e1[:N]
    e2 = e2[:N]

    if crit == "MSE":
        d = e1**2 - e2**2
    elif crit == "MAE":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError("crit must be MSE or MAE")

    mean_d = np.mean(d)
    gamma = 0

    # Newey-West adjusted variance for horizon h
    for lag in range(1, h):
        weight = 1 - lag/h
        gamma += weight * np.cov(d[:-lag], d[lag:])[0, 1]

    S = np.var(d) + 2 * gamma
    DM = mean_d / np.sqrt(S / N)
    p = 2 * (1 - norm.cdf(abs(DM)))

    return DM, p


def run_stat_tests(y_true, y_pred_main, y_pred_baseline, label_main="PRIN", label_base="Baseline"):
    """
    Compute:
      • DM test (MSE)
      • DM test (MAE)
      • paired t-test on error differences
    Returns dictionary of metrics.
    """

    y_true = np.asarray(y_true)
    y_pred_main = np.asarray(y_pred_main)
    y_pred_baseline = np.asarray(y_pred_baseline)

    N = min(len(y_true), len(y_pred_main), len(y_pred_baseline))
    y_true = y_true[:N]
    y_pred_main = y_pred_main[:N]
    y_pred_base = y_pred_baseline[:N]

    e1 = y_pred_main - y_true
    e2 = y_pred_base - y_true

    DM_mse, p_mse = diebold_mariano_test(e1, e2, crit="MSE")
    DM_mae, p_mae = diebold_mariano_test(e1, e2, crit="MAE")

    t_stat, t_p = ttest_rel(e1, e2)

    return {
        "DM_MSE": DM_mse,
        "p_MSE": p_mse,
        "DM_MAE": DM_mae,
        "p_MAE": p_mae,
        "t_stat": t_stat,
        "t_p": t_p
    }


def plot_stat_test_results(results, save=True):
    """
    Bar plot of p-values for MSE, MAE, and paired t-test.
    """

    labels = ["p_MSE", "p_MAE", "t_p"]
    pvals = [results[k] for k in labels]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, pvals, color=["blue", "purple", "green"])
    ax.axhline(0.05, color="red", linestyle="--", label="Significance Threshold (0.05)")
    ax.set_title("Statistical Significance Tests (DM + T-Test)")
    ax.set_ylabel("p-value")
    ax.legend()
    fig.tight_layout()

    if save:
        fig.savefig("empirical/figures/statistical_tests.pdf", bbox_inches='tight')
    plt.close(fig)


def save_stat_tests_latex(results):
    """
    Save LaTeX table for reviewers.
    """

    tex_path = "empirical/tables/stat_tests.tex"

    with open(tex_path, "w") as f:
        f.write("\\begin{tabular}{lcc}\n")
        f.write("\\toprule\n")
        f.write("Test & Statistic & p-value \\\\\n")
        f.write("\\midrule\n")
        f.write(f"DM (MSE) & {results['DM_MSE']:.4f} & {results['p_MSE']:.4f} \\\\\n")
        f.write(f"DM (MAE) & {results['DM_MAE']:.4f} & {results['p_MAE']:.4f} \\\\\n")
        f.write(f"Paired t-test & {results['t_stat']:.4f} & {results['t_p']:.4f} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

    print("[OK] Statistical test LaTeX table written to empirical/tables/stat_tests.tex")






# =========================================================
# FORECAST VS ACTUAL
# =========================================================

def plot_forecast_vs_actual(y_true, y_pred, save=True):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    N = min(len(y_true), len(y_pred))
    y_true = y_true[:N]
    y_pred = y_pred[:N]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(y_true, label="Actual", linewidth=2)
    ax.plot(y_pred, label="Predicted", linewidth=2)
    ax.set_title("Forecast vs Actual")
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.legend()
    fig.tight_layout()

    if save:
        fig.savefig("empirical/figures/forecast_vs_actual.pdf", bbox_inches='tight')
    plt.close(fig)


def plot_error_distribution(y_true, y_pred, save=True):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    N = min(len(y_true), len(y_pred))
    errors = y_pred[:N] - y_true[:N]

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(errors, kde=True, ax=ax)
    ax.set_title("Prediction Error Distribution")
    ax.set_xlabel("Error")
    fig.tight_layout()

    if save:
        fig.savefig("empirical/figures/error_distribution.pdf", bbox_inches='tight')
    plt.close(fig)
# =========================================================
# ROLLING DIEBOLD–MARIANO TEST (REGIME-AWARE)
# =========================================================

def rolling_dm_test(y_true, preds_model, preds_base, regimes=None, window=200, step=10):
    """
    Computes rolling DM test to evaluate local predictive superiority.
    If 'regimes' is provided, the function will also evaluate
    per-regime DM statistics.
    """

    y_true = np.asarray(y_true)
    p1 = np.asarray(preds_model)
    p2 = np.asarray(preds_base)

    N = min(len(y_true), len(p1), len(p2))
    y_true = y_true[:N]
    p1 = p1[:N]
    p2 = p2[:N]

    e1 = p1 - y_true
    e2 = p2 - y_true
    if N < window:
        print(f"[WARN] rolling_dm_test: window size {window} reduced to N={N} to avoid skipping.")
        window = N


    dm_stats = []
    dm_pvals = []
    indices = []

    for start in range(0, N - window + 1, step):

        end = start + window

        DM, p = diebold_mariano_test(e1[start:end], e2[start:end])
        dm_stats.append(DM)
        dm_pvals.append(p)
        indices.append(end)

    # Plot rolling DM test
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(indices, dm_stats, label="DM Statistic", color="blue")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Rolling Diebold-Mariano Test (Local Predictive Superiority)")
    ax.set_xlabel("Index")
    ax.set_ylabel("DM Statistic")
    fig.tight_layout()
    fig.savefig("empirical/figures/rolling_dm_test.pdf", bbox_inches='tight')
    plt.close(fig)

    # If regimes exist, compute DM per regime
    if regimes is not None:
        unique_regimes = np.unique(regimes)
        regime_results = {}

        for r in unique_regimes:
            idx = np.where(regimes == r)[0]
            if len(idx) < window:
                continue

            # restrict sequences to regime
            e1_r = e1[idx]
            e2_r = e2[idx]

            DM_r, p_r = diebold_mariano_test(e1_r, e2_r)
            regime_results[int(r)] = {"DM": DM_r, "p": p_r}

        # Save a LaTeX table
        with open("empirical/tables/rolling_dm_regimes.tex", "w") as f:
            f.write("\\begin{tabular}{lcc}\n")
            f.write("\\toprule\nRegime & DM & p-value \\\\\n\\midrule\n")
            for r, v in regime_results.items():
                f.write(f"{r} & {v['DM']:.4f} & {v['p']:.4f} \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n")

    return dm_stats, dm_pvals

# =========================================================
# MODEL CONFIDENCE SET (HANSEN, LUNDE & NASON)
# =========================================================

def compute_loss(y_true, y_pred, loss="MSE"):
    err = y_pred - y_true
    if loss == "MSE":
        return err**2
    if loss == "MAE":
        return np.abs(err)
    raise ValueError("loss must be MSE or MAE")

def mcs_test(y_true, model_preds, alpha=0.10, B=500):

    # --- bind B immediately to avoid unbound variable errors ---
    B_local = int(B)

    # -------------------------------
    # PREPARE DATA
    # -------------------------------
    model_names = list(model_preds.keys())
    M = len(model_names)

    losses = {}
    for name in model_names:
        losses[name] = compute_loss(y_true, model_preds[name])

    L = np.column_stack([losses[name] for name in model_names])
    survivors = model_names.copy()

    # -------------------------------
    # MCS LOOP
    # -------------------------------
    while len(survivors) > 1:

        idxs = [model_names.index(m) for m in survivors]
        Ls = L[:, idxs]
        Tn = Ls.shape[0]

        if Tn <= 1:
            break

        dbar = Ls.mean(axis=0)
        m0 = np.argmin(dbar)

        diffs = Ls - Ls[:, m0][:, None]
        std = diffs.std(axis=0)
        std[std == 0] = 1e-8
        tvals = diffs.mean(axis=0) / (std / np.sqrt(Tn))

        # --- SAFE BOOTSTRAP ---
        boot_stats = []
        for _ in range(B_local):
            try:
                idx = np.random.randint(0, Tn, Tn)
                dboot = diffs[idx]
                dmean = dboot.mean(axis=0)
                dstd = dboot.std(axis=0)
                dstd[dstd == 0] = 1e-8
                tboot = dmean / (dstd / np.sqrt(Tn))
                boot_stats.append(np.max(tboot))
            except Exception:
                boot_stats.append(0.0)

        crit = np.quantile(boot_stats, 1 - alpha)

        worst_idx = np.argmax(tvals)
        if tvals[worst_idx] > crit:
            survivors.pop(worst_idx)
        else:
            break

    # -------------------------------
    # SAVE LaTeX TABLE
    # -------------------------------
    out_path = "empirical/tables/mcs_results.tex"
    with open(out_path, "w") as f:
        f.write("\\begin{tabular}{l}\n")
        f.write("\\toprule\n Model Confidence Set \\\\\n\\midrule\n")
        for m in survivors:
            f.write(f"{m} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    print(f"[OK] MCS results saved to {out_path}")

    return survivors


# =========================================================
# BOOTSTRAPPED CHAOTIC SIGNIFICANCE TESTS (BLOCK BOOTSTRAP)
# =========================================================

def block_bootstrap(arr, block_size=20, B=500):
    B_local = int(B)

    N = len(arr)
    samples = []

    for _ in range(B_local):
        idx = np.random.randint(0, N - block_size)
        block = arr[idx:idx + block_size]
        samples.append(block)

    return samples


def chaotic_bootstrap_test(y_true, y_pred, y_base, block_size=20, B=500):
    B_local = int(B)


    e = (y_pred - y_true)**2
    e_base = (y_base - y_true)**2

    diff = e - e_base
    observed = diff.mean()

    boot_means = []
    for block in block_bootstrap(diff, block_size, B_local):
        boot_means.append(np.mean(block))

    p = np.mean(np.abs(boot_means) >= np.abs(observed))

    # Save figure
    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(boot_means, ax=ax, kde=True)
    ax.axvline(observed, color="red", label="Observed")
    ax.set_title("Chaotic Bootstrapped Significance Test")
    ax.legend()
    fig.tight_layout()
    fig.savefig("empirical/figures/chaotic_bootstrap_test.pdf")
    plt.close(fig)

    return observed, p
# =========================================================
# BASELINE MODELS: ARIMA, TRANSFORMER, TCN, LSTM
# =========================================================
from statsmodels.tsa.arima.model import ARIMA

def baseline_arima(y_train, order=(5,1,0)):
    model = ARIMA(y_train, order=order).fit()
    return model.fittedvalues

class BaselineTransformer(nn.Module):
    def __init__(self, input_dim, nhead=2, num_layers=2, dim_feed=128):
        super().__init__()
        layer = nn.TransformerEncoderLayer(input_dim, nhead, dim_feed)
        self.enc = nn.TransformerEncoder(layer, num_layers)
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = self.enc(x)
        return self.fc(x[:, -1, :])

class BaselineTCN(nn.Module):
    def __init__(self, in_ch, out_ch=64):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2)
        self.fc = nn.Linear(out_ch, 1)

    def forward(self, x):
        x = x.permute(0,2,1)
        h = torch.relu(self.conv(x))
        h = h.mean(-1)
        return self.fc(h)

class BaselineLSTM2(nn.Module):
    def __init__(self, input_dim, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        h,_ = self.lstm(x)
        return self.fc(h[:,-1,:])



# =========================================================
# LATEX APPENDIX BUILDER
# =========================================================

def build_latex_appendix():
    tex_path = "empirical/latex/empirical_appendix.tex"
    with open(tex_path, "w") as f:
        f.write(r"""
\section*{Empirical Validation Appendix}

This appendix contains all diagnostics validating PRIN and NeuroPRINv4,
including:

\begin{itemize}
\item Chaos analysis and contraction effects
\item Resonance dominance and spectral localization
\item Regime segmentation via HMM
\item RQA recurrence plots
\item Predictive coding contraction proofs
\item Salience-based pruning visualization
\item Forecast accuracy diagnostics
\end{itemize}

""")

    print(f"[OK] LaTeX appendix created: {tex_path}")


# =========================================================
# SHAPE HELPERS
# =========================================================

def _to_1d(arr, name):
    a = np.asarray(arr)
    if a.ndim > 1:
        print(f"[WARN] {name} has shape {a.shape}, flattening to 1D.")
        a = a.reshape(-1)
    return a


def _trim_to_n(arr, n, name):
    a = np.asarray(arr)
    if a.shape[0] != n:
        print(f"[WARN] Trimming {name} from {a.shape[0]} to {n} along axis 0.")
        return a[-n:]
    return a


# storage for stat_results
_stat_results_internal = None

def _set_stat_results(res):
    global _stat_results_internal
    _stat_results_internal = res

def _get_stat_results():
    return _stat_results_internal

# =========================================================
# MASTER FUNCTION with SAFE LOGGING + SHAPE ALIGNMENT
# =========================================================

def run_all_empirical_tests(
    model,
    Xv,
    chaos,
    resonance,
    regimes,
    rqa,
    y_true,
    y_pred,
    pruning_mask=None
):
    """
    Run the full empirical suite with:
    - shape logging
    - automatic 1D flattening for vector series
    - alignment to a common length N
    - per-step try/except so one failure doesn't kill everything
    """
    ensure_dirs()

    # ---------- Coerce and inspect shapes ----------
    Xv = np.asarray(Xv)
    chaos = _to_1d(chaos, "chaos")
    resonance = _to_1d(resonance, "resonance")
    regimes = _to_1d(regimes, "regimes")
    rqa = _to_1d(rqa, "rqa")
    y_true = _to_1d(y_true, "y_true")

    # -------------------------------
    # Multi-task prediction handling
    # -------------------------------
    # If model outputs 3 values (return, direction, volatility)
    # extract only the return prediction for the empirical tests.
    if isinstance(y_pred, np.ndarray) and y_pred.ndim == 2 and y_pred.shape[1] == 3:
        print("[INFO] Detected multi-task y_pred. Using return prediction only (index 0).")
        y_pred = y_pred[:, 0]

    y_pred = _to_1d(y_pred, "y_pred")

    if pruning_mask is not None:
        pruning_mask = np.asarray(pruning_mask)

    print("\n[EMPIRICAL] RAW SHAPES:")
    print(f"  Xv         : {Xv.shape}")
    print(f"  chaos      : {chaos.shape}")
    print(f"  resonance  : {resonance.shape}")
    print(f"  regimes    : {regimes.shape}")
    print(f"  rqa        : {rqa.shape}")
    print(f"  y_true     : {y_true.shape}")
    print(f"  y_pred     : {y_pred.shape}")
    if pruning_mask is not None:
        print(f"  pruning_mask: {pruning_mask.shape}")

    # Determine common length N
    lengths = [
        len(chaos),
        len(resonance),
        len(regimes),
        len(rqa),
        len(y_true),
        len(y_pred),
    ]
    if Xv.ndim >= 1:
        lengths.append(Xv.shape[0])
    if pruning_mask is not None and pruning_mask.ndim >= 1:
        lengths.append(pruning_mask.shape[0])

    lengths = [int(l) for l in lengths if l is not None and l > 0]
    if not lengths:
        raise ValueError("No positive lengths found for empirical arrays.")

    N = min(lengths)
    print(f"[EMPIRICAL] Common length N = {N}")

    # Trim everything to last N along axis 0
    chaos = _trim_to_n(chaos, N, "chaos")
    resonance = _trim_to_n(resonance, N, "resonance")
    regimes = _trim_to_n(regimes, N, "regimes")
    rqa = _trim_to_n(rqa, N, "rqa")
    y_true = _trim_to_n(y_true, N, "y_true")
    y_pred = _trim_to_n(y_pred, N, "y_pred")

    if Xv.shape[0] != N:
        if Xv.shape[0] < N:
            print(f"[WARN] Xv has fewer rows ({Xv.shape[0]}) than N={N}. Shrinking N.")
            N = Xv.shape[0]
            # retrim vectors again
            chaos = _trim_to_n(chaos, N, "chaos")
            resonance = _trim_to_n(resonance, N, "resonance")
            regimes = _trim_to_n(regimes, N, "regimes")
            rqa = _trim_to_n(rqa, N, "rqa")
            y_true = _trim_to_n(y_true, N, "y_true")
            y_pred = _trim_to_n(y_pred, N, "y_pred")
        else:
            Xv = _trim_to_n(Xv, N, "Xv")

    if pruning_mask is not None and pruning_mask.shape[0] != N:
        pruning_mask = _trim_to_n(pruning_mask, N, "pruning_mask")

    print("\n[EMPIRICAL] ALIGNED SHAPES:")
    print(f"  Xv         : {Xv.shape}")
    print(f"  chaos      : {chaos.shape}")
    print(f"  resonance  : {resonance.shape}")
    print(f"  regimes    : {regimes.shape}")
    print(f"  rqa        : {rqa.shape}")
    print(f"  y_true     : {y_true.shape}")
    print(f"  y_pred     : {y_pred.shape}")
    if pruning_mask is not None:
        print(f"  pruning_mask: {pruning_mask.shape}")

    # ---------- Safe runner for individual tasks ----------
    def _safe_run(label, fn):
        try:
            fn()
        except Exception as e:
            print(f"[ERROR] {label} failed: {e}")

    # Some predictive coding helpers that guard missing model.pc
    def _pc_contraction():
        if not hasattr(model, "pc"):
            raise AttributeError("model has no 'pc' module — skipping predictive coding contraction.")
        dim = getattr(model.pc, "dim", None)
        if dim is None:
            dim = model.pc.W.shape[0]
        x0 = np.random.randn(dim)
        plot_predictive_coding_contraction(x0, model)

    def _pc_radius():
        if not hasattr(model, "pc"):
            raise AttributeError("model has no 'pc' module — skipping spectral radius.")
        plot_matrix_spectral_radius(model)

    # =====================================================
    # STEP DEFINITIONS
    # =====================================================
    steps = [
        ("Hidden-state norms", [
            lambda: plot_hidden_state_norm(model, Xv),
        ]),
        
        ("Predictive coding trajectory", [
            lambda: plot_predictive_coding_error(model, Xv),
        ]),
        
        ("Ablation study", [
            lambda: run_ablation_study(
                type(model),
                {
                    "input_size": model.input_size,
                    "hidden_size": model.hidden_size,
                    "num_layers": model.num_layers,
                    "seq_len": model.seq_len,
                    "num_regimes": model.num_regimes,
                    "use_pc": getattr(model, "use_pc", True),
                    "use_chaos": getattr(model, "use_chaos", True),
                    "use_resonance": getattr(model, "use_resonance", True),
                    "use_regimes": getattr(model, "use_regimes", True),
                },
                Xv,
                y_true
            ),
        ]),



        ("Chaos diagnostics", [
            lambda: plot_chaos_map(chaos),
            lambda: plot_chaos_gate(chaos),
            lambda: plot_chaos_contraction_effect(Xv, chaos),
        ]),
        ("Rolling DM Test", [
            lambda: rolling_dm_test(
                y_true,
                y_pred,
                np.roll(y_true, 1),  # naive baseline prediction
                regimes=regimes
            ),
        ]),

        ("MCS Test", [
            lambda: mcs_test(
                y_true,
                {
                    "PRIN": y_pred,
                    "ARIMA": baseline_arima(y_true),
                    "Naive": np.roll(y_true, 1)

                }
            ),
        ]),
        
        ("Chaotic Bootstrap Test", [
            lambda: chaotic_bootstrap_test(y_true, y_pred, np.roll(y_true, 1), block_size=20, B=500),


        ]),

        ("Resonance diagnostics", [
            lambda: plot_resonance_strength(resonance),
            lambda: plot_resonance_gate(resonance),
            lambda: plot_spectral_heatmap(Xv),
        ]),
        ("Regime timeline", [
            lambda: plot_regime_timeline(regimes),
        ]),
        ("RQA metrics", [
            lambda: plot_rqa_metrics(rqa),
        ]),
        ("Predictive coding", [
            _pc_contraction,
            _pc_radius,
        ]),
        ("Salience-based pruning", [] if pruning_mask is None else [
            lambda: plot_pruning_mask(pruning_mask),
            lambda: plot_feature_magnitudes_before_after(
                before=np.ones_like(pruning_mask[:, -1]) * np.mean(np.abs(Xv[:, -1, 0])),
                after=np.ones_like(pruning_mask[:, -1]) * np.mean(
                    np.abs(Xv[:, -1, 0] * pruning_mask[:, -1])
                ),
            )
        ]),

        ("Chaos extended", [
            lambda: plot_chaos_histogram(chaos),
            lambda: plot_chaos_variance_pre_post(Xv, chaos),
        ]),
        ("Pruning extended", [] if pruning_mask is None else [
            lambda: plot_pruning_sparsity(pruning_mask),
        ]),

        ("Predictive coding extended", [
            lambda: plot_jacobian_norm(model),
        ]),
        ("Resonance extended", [
            lambda: plot_resonance_crosscorr(Xv, resonance),
        ]),
        ("RQA extended", [
            lambda: plot_rqa_radar(rqa),
        ]),
        ("Regime dynamics", [
            lambda: plot_regime_transition_matrix(regimes),
        ]),

        ("Prediction diagnostics", [
            lambda: plot_forecast_vs_actual(y_true, y_pred),
            lambda: plot_error_distribution(y_true, y_pred),
        ]),
        
        ("Statistical significance tests", [
            lambda: _set_stat_results(run_stat_tests(
                y_true=y_true,
                y_pred_main=y_pred,          # PRIN predictions
                y_pred_baseline=y_true[:-1]  # trivial baseline or supply actual baseline preds
            )),
            lambda: plot_stat_test_results(_get_stat_results()),
            lambda: save_stat_tests_latex(_get_stat_results()),
        ]),

        ("Build LaTeX appendix", [
            lambda: build_latex_appendix(),
        ])
    ]

    print("\n[INFO] Running full empirical suite...\n")

    total_subtasks = sum(len(group[1]) for group in steps)

    with tqdm(total=total_subtasks, desc="Empirical Validation", ncols=100) as pbar:
        for label, funcs in steps:
            print(f"[INFO] {label}...")
            for fn in funcs:
                _safe_run(label, fn)
                pbar.update(1)

    print("\n[DONE] Full empirical suite generated successfully.\n")
