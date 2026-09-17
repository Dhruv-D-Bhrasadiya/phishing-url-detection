"""
experiments.py
--------------
Orchestrates the full experimental pipeline:

  Experiment 1 -- In-distribution (random 80/20 split)
    1a. D1 only
    1b. D2 only

  Experiment 2 -- Cross-dataset (generalisation test)
    2a. Train on D1 -> Test on D2
    2b. Train on D2 -> Test on D1

  Bonus: Feature-selection mitigation (low JS-divergence features only)
  Bonus: Data-blending (mix 10/20/30% of target into training)

Usage
-----
  # Full run with real data
  python src/experiments.py \\
      --d1 data/raw/dataset_phishing_uci.csv \\
      --d2 data/raw/phiusiil_phishing_url.csv

  # Smoke-test (synthetic data, fast)
  python src/experiments.py --quick-check

  # Skip SVM (slow) and run only RF + XGBoost
  python src/experiments.py --d1 ... --d2 ... --models rf xgb

  # Limit dataset size for testing
  python src/experiments.py --d1 ... --d2 ... --d1-sample 2000 --d2-sample 5000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split

from src.data_loader import load_both, generate_synthetic_dataset
from src.evaluation import (
    compute_metrics,
    print_report,
    save_metrics,
    plot_confusion_matrix,
    plot_roc_pr_curves,
    plot_feature_importance,
    plot_shap,
    compute_distribution_shift,
    plot_distribution_shift,
    plot_shift_heatmap,
    plot_generalisation_gap,
    plot_metrics_heatmap,
)
from src.features import FEATURE_NAMES
from src.models import get_model, MODEL_NAMES, PARAM_GRIDS, ALL_MODELS

RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)


def _get_proba(model, X: pd.DataFrame) -> np.ndarray:
    """Get positive-class probability, with fallback for models without predict_proba."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        # Normalise to [0,1]
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
    return model.predict(X).astype(float)


def _tune_and_fit(
    model_key: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 20,
    cv: int = 5,
    tune: bool = True,
) -> object:
    """Optionally run RandomizedSearchCV then refit on full training set."""
    base_model = get_model(model_key)

    if not tune:
        print(f"  Fitting {MODEL_NAMES[model_key]} (no tuning)...")
        base_model.fit(X_train, y_train)
        return base_model

    grid = PARAM_GRIDS.get(model_key, {})
    if not grid:
        print(f"  No param grid for {model_key}, fitting with defaults...")
        base_model.fit(X_train, y_train)
        return base_model

    print(f"  Tuning {MODEL_NAMES[model_key]} ({n_iter} iter, {cv}-fold CV)...")
    search = RandomizedSearchCV(
        base_model,
        param_distributions=grid,
        n_iter=n_iter,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring="f1",
        n_jobs=-1,
        random_state=42,
        verbose=0,
    )
    search.fit(X_train, y_train)
    print(f"    Best params: {search.best_params_}")
    print(f"    CV F1: {search.best_score_:.4f}")
    return search.best_estimator_


def _run_single_experiment(
    model_key: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    exp_label: str,
    tune: bool = True,
    n_iter: int = 20,
) -> dict:
    """Train, predict, compute metrics for one model × one experiment."""
    t0 = time.time()
    model = _tune_and_fit(model_key, X_train, y_train, n_iter=n_iter, tune=tune)
    train_time = time.time() - t0

    y_pred = model.predict(X_test)
    y_prob = _get_proba(model, X_test)
    metrics = compute_metrics(y_test, y_pred, y_prob)
    metrics["train_time_s"] = round(train_time, 2)

    # Save confusion matrix
    cm_path = FIGURES_DIR / f"cm_{exp_label}_{model_key}.png"
    plot_confusion_matrix(
        y_test, y_pred,
        title=f"{MODEL_NAMES[model_key]} -- {exp_label}",
        out_path=cm_path,
    )

    return {"model": model, "metrics": metrics, "y_pred": y_pred, "y_prob": y_prob}


# ---------------------------------------------------------------------------
# Experiment 1 -- In-distribution random split
# ---------------------------------------------------------------------------

def experiment_1(
    X1: pd.DataFrame, y1: pd.Series,
    X2: pd.DataFrame, y2: pd.Series,
    model_keys: list[str],
    tune: bool = True,
    n_iter: int = 20,
) -> dict[str, dict]:
    """Random 80/20 split on D1 and D2 separately."""
    print("\n" + "="*60)
    print("  EXPERIMENT 1 -- In-distribution (random split)")
    print("="*60)

    all_results = {}

    for dataset_name, X, y in [("D1-UCI", X1, y1), ("D2-PhiUSIIL", X2, y2)]:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        print(f"\n[{dataset_name}] Train: {len(X_tr):,}  Test: {len(X_te):,}")

        roc_items = []
        for mk in model_keys:
            exp_label = f"exp1_{dataset_name}_{mk}"
            print(f"\n  -> {MODEL_NAMES[mk]}")
            out = _run_single_experiment(mk, X_tr, y_tr, X_te, y_te,
                                          exp_label, tune=tune, n_iter=n_iter)
            key = f"{MODEL_NAMES[mk]} -- {dataset_name} (random split)"
            all_results[key] = out["metrics"]
            roc_items.append({
                "name": MODEL_NAMES[mk],
                "y_true": y_te,
                "y_prob": out["y_prob"],
            })

            # Feature importance
            plot_feature_importance(
                out["model"], FEATURE_NAMES,
                model_name=f"{MODEL_NAMES[mk]} ({dataset_name})",
                out_path=FIGURES_DIR / f"feat_imp_{mk}_{dataset_name}_exp1.png",
            )

        # ROC/PR curves for all models on this dataset
        plot_roc_pr_curves(
            roc_items,
            title_prefix=f"Exp1 {dataset_name}",
            out_path=FIGURES_DIR / f"roc_pr_{dataset_name}_exp1.png",
        )

    return all_results


# ---------------------------------------------------------------------------
# Experiment 2 -- Cross-dataset generalisation
# ---------------------------------------------------------------------------

def experiment_2(
    X1: pd.DataFrame, y1: pd.Series,
    X2: pd.DataFrame, y2: pd.Series,
    model_keys: list[str],
    tune: bool = True,
    n_iter: int = 20,
) -> dict[str, dict]:
    """Train on one dataset, test on the other (both directions)."""
    print("\n" + "="*60)
    print("  EXPERIMENT 2 -- Cross-dataset (generalisation test)")
    print("="*60)

    all_results = {}
    # Each tuple: (display_label, file_slug, X_train, y_train, X_test, y_test)
    transfer_pairs = [
        ("D1-UCI -> D2-PhiUSIIL", "D1UCI_to_D2PhiUSIIL", X1, y1, X2, y2),
        ("D2-PhiUSIIL -> D1-UCI", "D2PhiUSIIL_to_D1UCI",  X2, y2, X1, y1),
    ]

    for pair_label, pair_slug, X_train, y_train, X_test, y_test in transfer_pairs:
        print(f"\n[{pair_label}] Train: {len(X_train):,}  Test: {len(X_test):,}")
        roc_items = []

        for mk in model_keys:
            exp_label = f"exp2_{mk}_{pair_slug}"
            print(f"\n  -> {MODEL_NAMES[mk]}")
            out = _run_single_experiment(
                mk, X_train, y_train, X_test, y_test,
                exp_label, tune=tune, n_iter=n_iter,
            )
            key = f"{MODEL_NAMES[mk]} -- {pair_label}"
            all_results[key] = out["metrics"]
            roc_items.append({
                "name": MODEL_NAMES[mk],
                "y_true": y_test,
                "y_prob": out["y_prob"],
            })

        plot_roc_pr_curves(
            roc_items,
            title_prefix=f"Exp2 {pair_label}",
            out_path=FIGURES_DIR / f"roc_pr_exp2_{pair_slug}.png",
        )

    return all_results


# ---------------------------------------------------------------------------
# Distribution shift analysis
# ---------------------------------------------------------------------------

def run_distribution_analysis(
    X1: pd.DataFrame,
    X2: pd.DataFrame,
) -> pd.DataFrame:
    print("\n" + "="*60)
    print("  DISTRIBUTION SHIFT ANALYSIS")
    print("="*60)
    shift_df = compute_distribution_shift(X1, X2)
    print("\nTop 10 most-shifted features (JS divergence):")
    print(shift_df.head(10).to_string(index=False))

    # Save CSV
    shift_df.to_csv(METRICS_DIR / "distribution_shift.csv", index=False)

    plot_distribution_shift(X1, X2, top_n=10,
                             out_path=FIGURES_DIR / "distribution_shift_violins.png")
    plot_shift_heatmap(shift_df,
                       out_path=FIGURES_DIR / "distribution_shift_heatmap.png")
    return shift_df


# ---------------------------------------------------------------------------
# Bonus: Feature-selection mitigation
# ---------------------------------------------------------------------------

def experiment_mitigation_feature_selection(
    X1: pd.DataFrame, y1: pd.Series,
    X2: pd.DataFrame, y2: pd.Series,
    shift_df: pd.DataFrame,
    js_threshold: float = 0.3,
    model_keys: list[str] | None = None,
    tune: bool = False,
) -> dict[str, dict]:
    """Retrain using only low-JS-divergence (stable) features."""
    print("\n" + "="*60)
    print("  BONUS: Feature-selection mitigation")
    print("="*60)

    stable_features = shift_df.loc[shift_df["js_divergence"] < js_threshold, "feature"].tolist()
    stable_features = [f for f in stable_features if f in X1.columns and f in X2.columns]
    print(f"  Using {len(stable_features)} stable features (JS < {js_threshold}): {stable_features}")

    if not stable_features:
        print("  [WARN] No stable features found. Skipping mitigation.")
        return {}

    if model_keys is None:
        model_keys = ["rf", "xgb"]

    results = {}
    for mk in model_keys:
        out = _run_single_experiment(
            mk,
            X1[stable_features], y1,
            X2[stable_features], y2,
            f"mitigation_featsel_{mk}",
            tune=tune,
        )
        key = f"{MODEL_NAMES[mk]} -- Feat-Select D1->D2"
        results[key] = out["metrics"]

    return results


# ---------------------------------------------------------------------------
# Bonus: Data-blending mitigation
# ---------------------------------------------------------------------------

def experiment_mitigation_blending(
    X1: pd.DataFrame, y1: pd.Series,
    X2: pd.DataFrame, y2: pd.Series,
    blend_ratios: list[float] | None = None,
    model_key: str = "rf",
    tune: bool = False,
) -> dict[str, dict]:
    """Mix α% of D2 into D1 training data and measure generalisation improvement."""
    print("\n" + "="*60)
    print("  BONUS: Data-blending mitigation")
    print("="*60)

    if blend_ratios is None:
        blend_ratios = [0.0, 0.1, 0.2, 0.3]

    # Hold out 20% of D2 for testing
    X2_tr, X2_te, y2_tr, y2_te = train_test_split(
        X2, y2, test_size=0.2, stratify=y2, random_state=42
    )

    results = {}
    for ratio in blend_ratios:
        n_blend = int(ratio * len(X1))
        if n_blend == 0:
            X_aug, y_aug = X1.copy(), y1.copy()
        else:
            X_blend = X2_tr.sample(min(n_blend, len(X2_tr)), random_state=42)
            y_blend = y2_tr.loc[X_blend.index]
            X_aug = pd.concat([X1, X_blend], ignore_index=True)
            y_aug = pd.concat([y1, y_blend], ignore_index=True)

        print(f"\n  Blend ratio={ratio:.0%}  Train size: {len(X_aug):,}")
        out = _run_single_experiment(
            model_key, X_aug, y_aug, X2_te, y2_te,
            f"blend_{int(ratio*100):02d}pct_{model_key}",
            tune=tune,
        )
        key = f"{MODEL_NAMES[model_key]} -- Blend {ratio:.0%} D2->D1"
        results[key] = out["metrics"]

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phishing URL Detection -- Generalisation Study"
    )
    parser.add_argument("--d1", type=str, default=None,
                        help="Path to UCI D1 CSV")
    parser.add_argument("--d2", type=str, default=None,
                        help="Path to PhiUSIIL D2 CSV")
    parser.add_argument("--d1-sample", type=int, default=None,
                        help="Max rows to use from D1 (for speed)")
    parser.add_argument("--d2-sample", type=int, default=None,
                        help="Max rows to use from D2 (for speed)")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS,
                        choices=ALL_MODELS,
                        help="Models to run (default: all)")
    parser.add_argument("--no-tune", action="store_true",
                        help="Skip hyperparameter tuning (faster)")
    parser.add_argument("--n-iter", type=int, default=20,
                        help="RandomizedSearchCV iterations")
    parser.add_argument("--quick-check", action="store_true",
                        help="Run on synthetic data for smoke-testing")
    parser.add_argument("--skip-bonus", action="store_true",
                        help="Skip mitigation experiments")
    args = parser.parse_args()

    _ensure_dirs()

    # ---- Load data ----
    if args.quick_check:
        print("\n[QUICK-CHECK] Generating synthetic dataset...")
        X1, y1 = generate_synthetic_dataset(n_phishing=400, n_legit=400)
        X2, y2 = generate_synthetic_dataset(n_phishing=300, n_legit=300, random_state=99)
        tune = False
        model_keys = ["rf", "xgb"]  # fast subset
    else:
        if not args.d1 or not args.d2:
            parser.error("--d1 and --d2 are required unless --quick-check is set.")
        X1, y1, X2, y2 = load_both(
            args.d1, args.d2,
            d1_sample=args.d1_sample,
            d2_sample=args.d2_sample,
        )
        tune = not args.no_tune
        model_keys = args.models

    all_results: dict[str, dict] = {}

    # ---- Experiment 1 ----
    res1 = experiment_1(X1, y1, X2, y2, model_keys, tune=tune, n_iter=args.n_iter)
    all_results.update(res1)

    # ---- Experiment 2 ----
    res2 = experiment_2(X1, y1, X2, y2, model_keys, tune=tune, n_iter=args.n_iter)
    all_results.update(res2)

    # ---- Distribution shift ----
    shift_df = run_distribution_analysis(X1, X2)

    # ---- SHAP for best tree models ----
    print("\n  Generating SHAP plots (RF and XGBoost on D1)...")
    X1_tr, _, y1_tr, _ = train_test_split(X1, y1, test_size=0.2, stratify=y1, random_state=42)
    for mk in [k for k in model_keys if k in ("rf", "xgb")]:
        shap_model = _tune_and_fit(mk, X1_tr, y1_tr, tune=False)
        plot_shap(
            shap_model, X1_tr.sample(min(500, len(X1_tr)), random_state=42),
            model_name=MODEL_NAMES[mk],
            out_path=FIGURES_DIR / f"shap_{mk}.png",
        )

    # ---- Bonus experiments ----
    if not args.skip_bonus and not args.quick_check:
        bonus_models = [k for k in model_keys if k in ("rf", "xgb")]
        res_fs = experiment_mitigation_feature_selection(
            X1, y1, X2, y2, shift_df, model_keys=bonus_models
        )
        all_results.update(res_fs)

        res_blend = experiment_mitigation_blending(X1, y1, X2, y2, model_key="rf")
        all_results.update(res_blend)

    # ---- Summary ----
    print_report(all_results, title="FULL RESULTS SUMMARY")
    save_metrics(all_results, METRICS_DIR, "all_metrics.json")

    # ---- Summary plots ----
    plot_generalisation_gap(all_results, metric="f1",
                             out_path=FIGURES_DIR / "generalisation_gap_f1.png")
    plot_metrics_heatmap(all_results,
                          out_path=FIGURES_DIR / "metrics_heatmap.png")

    print("\n✓ All experiments complete.")
    print(f"  Figures -> {FIGURES_DIR}/")
    print(f"  Metrics -> {METRICS_DIR}/")


if __name__ == "__main__":
    main()
