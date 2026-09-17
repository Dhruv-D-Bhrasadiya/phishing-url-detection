"""
evaluation.py
-------------
Unified metric computation and visualisation for all experiments.

Functions
---------
compute_metrics(y_true, y_pred, y_prob)   -> dict of scalar metrics
print_report(results)                      -> pretty console table
plot_confusion_matrix(y_true, y_pred, ...) -> saves PNG
plot_roc_pr_curves(y_true, y_prob, ...)    -> saves PNG
plot_feature_importance(model, ...)        -> saves PNG (RF/XGB/LGBM)
plot_shap(model, X, ...)                   -> saves SHAP beeswarm PNG
plot_distribution_shift(X1, X2, ...)       -> saves violin/KDE plot PNG
plot_generalisation_gap(results_dict, ...) -> saves bar chart PNG
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script mode
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import jensenshannon
from scipy.special import rel_entr
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    classification_report,
)

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------

PALETTE = {
    "phishing": "#e63946",
    "legit":    "#2ec4b6",
    "rf":       "#4361ee",
    "xgb":      "#f77f00",
    "lgbm":     "#7209b7",
    "svm":      "#06d6a0",
    "gap":      "#e63946",
}

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

MODEL_COLORS = [PALETTE["rf"], PALETTE["xgb"], PALETTE["lgbm"], PALETTE["svm"]]


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_prob: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute all evaluation metrics and return as a flat dict."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    metrics: dict[str, float] = {
        "accuracy":         accuracy_score(y_true, y_pred),
        "precision":        precision_score(y_true, y_pred, zero_division=0),
        "recall":           recall_score(y_true, y_pred, zero_division=0),
        "f1":               f1_score(y_true, y_pred, zero_division=0),
        "f1_macro":         f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted":      f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "false_positive_rate": fpr,
        "tp": float(tp), "tn": float(tn), "fp": float(fp), "fn": float(fn),
    }

    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        metrics["pr_auc"]  = average_precision_score(y_true, y_prob)

    return metrics


def print_report(
    results: dict[str, dict[str, float]],
    title: str = "Results",
) -> None:
    """Pretty-print a table of metrics for multiple experiments."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "false_positive_rate"]
    header = f"{'Experiment':<35}" + "".join(f"{c:>12}" for c in cols)
    print(header)
    print("-" * len(header))
    for exp_name, m in results.items():
        row = f"{exp_name:<35}"
        for c in cols:
            val = m.get(c, float("nan"))
            row += f"{val:>12.4f}"
        print(row)
    print()


def save_metrics(
    results: dict[str, dict[str, float]],
    out_dir: str | Path,
    filename: str = "metrics.json",
) -> None:
    """Serialise results dict to JSON."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Metrics saved -> {out_dir / filename}")


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    out_path: str | Path | None = None,
) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm_pct,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn_r",
        xticklabels=["Legit", "Phishing"],
        yticklabels=["Legit", "Phishing"],
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": "%"},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontweight="bold")

    # Overlay raw counts
    for i in range(2):
        for j in range(2):
            ax.text(
                j + 0.5, i + 0.72,
                f"n={cm[i,j]:,}",
                ha="center", va="center",
                fontsize=9, color="black", alpha=0.7,
            )

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Confusion matrix saved -> {out_path}")
    return fig


# ---------------------------------------------------------------------------
# ROC & PR curves
# ---------------------------------------------------------------------------

def plot_roc_pr_curves(
    results_with_probs: list[dict],
    title_prefix: str = "",
    out_path: str | Path | None = None,
) -> plt.Figure:
    """
    Parameters
    ----------
    results_with_probs : list of dicts, each with keys:
        'name', 'y_true', 'y_prob', 'color' (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, item in enumerate(results_with_probs):
        color = item.get("color", MODEL_COLORS[i % len(MODEL_COLORS)])
        y_true = np.asarray(item["y_true"])
        y_prob = np.asarray(item["y_prob"])

        # ROC
        fpr_arr, tpr_arr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        axes[0].plot(fpr_arr, tpr_arr, color=color, lw=2,
                     label=f"{item['name']} (AUC={auc:.3f})")

        # PR
        prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        axes[1].plot(rec_arr, prec_arr, color=color, lw=2,
                     label=f"{item['name']} (AP={ap:.3f})")

    # ROC decorations
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title(f"{title_prefix} ROC Curve", fontweight="bold")
    axes[0].legend(loc="lower right", fontsize=9)

    # PR decorations
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"{title_prefix} Precision-Recall Curve", fontweight="bold")
    axes[1].legend(loc="lower left", fontsize=9)

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"ROC/PR curves saved -> {out_path}")
    return fig


# ---------------------------------------------------------------------------
# Feature importance (tree models)
# ---------------------------------------------------------------------------

def plot_feature_importance(
    model,
    feature_names: list[str],
    model_name: str = "Model",
    top_n: int = 20,
    out_path: str | Path | None = None,
) -> plt.Figure:
    """Bar chart of feature importances for tree-based models."""
    # Handle Pipeline (e.g., SVM with scaler) -- no native feature importance
    if hasattr(model, "named_steps"):
        clf = model.named_steps.get("clf")
        if clf is None or not hasattr(clf, "feature_importances_"):
            print(f"[WARN] {model_name} does not support feature_importances_. Skipping.")
            return None
        importances = clf.feature_importances_
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        print(f"[WARN] {model_name} does not have feature_importances_. Skipping.")
        return None

    df_imp = pd.DataFrame({"feature": feature_names, "importance": importances})
    df_imp = df_imp.nlargest(top_n, "importance").sort_values("importance")

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    bars = ax.barh(df_imp["feature"], df_imp["importance"], color=PALETTE["rf"], edgecolor="white")
    ax.set_xlabel("Feature Importance (Gini / Gain)")
    ax.set_title(f"{model_name} -- Top {top_n} Feature Importances", fontweight="bold")

    # Annotate bars
    for bar, val in zip(bars, df_imp["importance"]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Feature importance saved -> {out_path}")
    return fig


# ---------------------------------------------------------------------------
# SHAP analysis
# ---------------------------------------------------------------------------

def plot_shap(
    model,
    X: pd.DataFrame,
    model_name: str = "Model",
    max_display: int = 20,
    out_path: str | Path | None = None,
) -> None:
    """Generate SHAP beeswarm plot for tree-based models."""
    try:
        import shap
    except ImportError:
        print("[WARN] shap not installed -- skipping SHAP plot.")
        return

    # Handle Pipeline (e.g. SVM with StandardScaler)
    clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model

    if not hasattr(clf, "feature_importances_"):
        print(f"[WARN] SHAP TreeExplainer not applicable for {model_name} -- no feature_importances_.")
        return

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X)

    # For binary classifiers shap_values may be a list [class0, class1]
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    # shap.summary_plot creates its own figure -- don't create one before it
    shap.summary_plot(sv, X, max_display=max_display, show=False)
    plt.title(f"{model_name} -- SHAP Feature Contributions", fontweight="bold")

    if out_path:
        plt.savefig(out_path, bbox_inches="tight")
        print(f"SHAP plot saved -> {out_path}")
    plt.close("all")


# ---------------------------------------------------------------------------
# Distribution shift analysis
# ---------------------------------------------------------------------------

def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """KL divergence D(P||Q) estimated from histograms."""
    if len(p) == 0 or len(q) == 0 or np.array_equal(p, q):
        return 0.0
    if p.min() == p.max() and q.min() == q.max() and p.min() == q.min():
        return 0.0
    combined = np.concatenate([p, q])
    if combined.min() == combined.max():
        return 0.0
    bins = np.histogram_bin_edges(combined, bins=50)
    if bins[0] == bins[-1]:
        return 0.0
    p_hist, _ = np.histogram(p, bins=bins)
    q_hist, _ = np.histogram(q, bins=bins)
    p_hist = p_hist.astype(float) + eps
    q_hist = q_hist.astype(float) + eps
    p_hist /= p_hist.sum()
    q_hist /= q_hist.sum()
    val = float(np.sum(rel_entr(p_hist, q_hist)))
    return max(0.0, val)


def _js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """Jensen-Shannon divergence (0=identical, 1=maximally different)."""
    if len(p) == 0 or len(q) == 0 or np.array_equal(p, q):
        return 0.0
    if p.min() == p.max() and q.min() == q.max() and p.min() == q.min():
        return 0.0
    combined = np.concatenate([p, q])
    if combined.min() == combined.max():
        return 0.0
    bins = np.histogram_bin_edges(combined, bins=50)
    if bins[0] == bins[-1]:
        return 0.0
    p_hist, _ = np.histogram(p, bins=bins)
    q_hist, _ = np.histogram(q, bins=bins)
    p_hist = p_hist.astype(float) + eps
    q_hist = q_hist.astype(float) + eps
    p_prob = p_hist / p_hist.sum()
    q_prob = q_hist / q_hist.sum()
    m = 0.5 * (p_prob + q_prob)
    js = 0.5 * (np.sum(rel_entr(p_prob, m)) + np.sum(rel_entr(q_prob, m)))
    return float(np.sqrt(max(0.0, float(js))))


def compute_distribution_shift(
    X1: pd.DataFrame,
    X2: pd.DataFrame,
) -> pd.DataFrame:
    """Compute KL and JS divergence for each feature between D1 and D2.

    Returns a DataFrame sorted by JS divergence (descending).
    """
    records = []
    for col in X1.columns:
        if col not in X2.columns:
            continue
        p = X1[col].dropna().values.astype(float)
        q = X2[col].dropna().values.astype(float)
        records.append({
            "feature": col,
            "kl_divergence": _kl_divergence(p, q),
            "js_divergence": _js_divergence(p, q),
            "mean_d1": p.mean(),
            "mean_d2": q.mean(),
            "std_d1":  p.std(),
            "std_d2":  q.std(),
        })
    df = pd.DataFrame(records).sort_values("js_divergence", ascending=False)
    return df.reset_index(drop=True)


def plot_distribution_shift(
    X1: pd.DataFrame,
    X2: pd.DataFrame,
    top_n: int = 10,
    out_path: str | Path | None = None,
) -> plt.Figure:
    """Violin plots of the top-N most-shifted features between D1 and D2."""
    shift_df = compute_distribution_shift(X1, X2)
    top_features = shift_df.head(top_n)["feature"].tolist()

    n_cols = 2
    n_rows = (top_n + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3.5))
    axes = axes.flatten()

    for idx, feat in enumerate(top_features):
        ax = axes[idx]
        data = pd.DataFrame({
            "value": np.concatenate([X1[feat].values, X2[feat].values]),
            "dataset": ["D1 (UCI)"] * len(X1) + ["D2 (PhiUSIIL)"] * len(X2),
        })
        sns.violinplot(
            data=data, x="dataset", y="value",
            hue="dataset",
            palette={"D1 (UCI)": PALETTE["rf"], "D2 (PhiUSIIL)": PALETTE["xgb"]},
            ax=ax, inner="quartile", cut=0, legend=False,
        )
        js = shift_df.loc[shift_df["feature"] == feat, "js_divergence"].values[0]
        ax.set_title(f"{feat}  (JS={js:.3f})", fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Value")

    # Hide unused axes
    for j in range(len(top_features), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distribution Shift: D1 vs D2", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Distribution shift plot saved -> {out_path}")
    return fig


def plot_shift_heatmap(
    shift_df: pd.DataFrame,
    out_path: str | Path | None = None,
) -> plt.Figure:
    """Bar chart of JS divergence per feature."""
    df = shift_df.sort_values("js_divergence", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(5, len(df) * 0.4)))
    colors = [PALETTE["gap"] if v > 0.3 else PALETTE["rf"] for v in df["js_divergence"]]
    ax.barh(df["feature"], df["js_divergence"], color=colors, edgecolor="white")
    ax.axvline(0.3, color="black", linestyle="--", linewidth=1, label="High-shift threshold (0.3)")
    ax.set_xlabel("Jensen-Shannon Divergence")
    ax.set_title("Feature Distribution Shift (D1 vs D2)\nRed = high-shift features", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Shift heatmap saved -> {out_path}")
    return fig


# ---------------------------------------------------------------------------
# Generalisation gap bar chart
# ---------------------------------------------------------------------------

def plot_generalisation_gap(
    results_dict: dict[str, dict[str, float]],
    metric: str = "f1",
    out_path: str | Path | None = None,
) -> plt.Figure:
    """
    Compare in-distribution vs cross-dataset F1 (or other metric) for each model.

    results_dict format:
    {
        "RF -- D1 random":     {"f1": 0.98, ...},
        "RF -- D1->D2 transfer": {"f1": 0.72, ...},
        ...
    }
    """
    # Group by model key
    models = ["Random Forest", "XGBoost", "LightGBM", "SVM (RBF)"]
    key_map = {
        "Random Forest": ("rf",  "RF"),
        "XGBoost":       ("xgb", "XGBoost"),
        "LightGBM":      ("lgbm","LightGBM"),
        "SVM (RBF)":     ("svm", "SVM"),
    }

    rows = []
    for exp_name, m in results_dict.items():
        rows.append({"experiment": exp_name, "value": m.get(metric, 0.0)})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(df))
    colors = []
    for name in df["experiment"]:
        if "random" in name.lower() or "in-dist" in name.lower():
            colors.append(PALETTE["lgbm"])
        else:
            colors.append(PALETTE["gap"])

    bars = ax.bar(x, df["value"], color=colors, edgecolor="white", width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(df["experiment"], rotation=35, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(metric.upper())
    ax.set_title(
        f"Generalisation Gap -- In-distribution (purple) vs Cross-dataset (red)\n{metric.upper()} score",
        fontweight="bold",
    )

    for bar, val in zip(bars, df["value"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.01,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=9,
        )

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Generalisation gap chart saved -> {out_path}")
    return fig


def plot_metrics_heatmap(
    results_dict: dict[str, dict[str, float]],
    metrics: list[str] | None = None,
    out_path: str | Path | None = None,
) -> plt.Figure:
    """Heatmap of all metrics for all experiments."""
    if metrics is None:
        metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "false_positive_rate"]

    rows = {}
    for exp, m in results_dict.items():
        rows[exp] = {met: m.get(met, float("nan")) for met in metrics}

    df = pd.DataFrame(rows).T
    fig, ax = plt.subplots(figsize=(max(10, len(metrics) * 1.5), max(6, len(rows) * 0.6)))
    sns.heatmap(
        df,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Full Metrics Heatmap -- All Experiments", fontweight="bold")
    ax.set_ylabel("")
    ax.set_xlabel("")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Metrics heatmap saved -> {out_path}")
    return fig
