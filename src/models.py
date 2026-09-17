"""
models.py
---------
Model factory + hyper-parameter search grids for three families:
  1. Random Forest   (sklearn)
  2. XGBoost         (xgboost)
  3. LightGBM        (lightgbm)
  4. SVM (RBF)       (sklearn)

Usage
-----
    from src.models import get_model, PARAM_GRIDS, MODEL_NAMES

    model = get_model("rf")
    grid  = PARAM_GRIDS["rf"]
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ---------------------------------------------------------------------------
# Model keys
# ---------------------------------------------------------------------------

MODEL_NAMES = {
    "rf":      "Random Forest",
    "xgb":     "XGBoost",
    "lgbm":    "LightGBM",
    "svm":     "SVM (RBF)",
}


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def get_model(key: str, **kwargs):
    """Return an unfitted estimator for the given model key.

    Parameters
    ----------
    key : str
        One of 'rf', 'xgb', 'lgbm', 'svm'.
    **kwargs
        Override default hyper-parameters.

    Returns
    -------
    sklearn-compatible estimator
    """
    defaults = _DEFAULTS.get(key)
    if defaults is None:
        raise ValueError(f"Unknown model key '{key}'. Choose from {list(MODEL_NAMES)}")
    params = {**defaults, **kwargs}

    if key == "rf":
        return RandomForestClassifier(**params)

    if key == "xgb":
        return XGBClassifier(**params)

    if key == "lgbm":
        return LGBMClassifier(**params)

    if key == "svm":
        # SVM needs feature scaling
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(**params)),
        ])

    raise ValueError(key)  # unreachable


# ---------------------------------------------------------------------------
# Default hyper-parameters (reasonable starting points)
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, dict] = {
    "rf": dict(
        n_estimators=300,
        max_depth=8,               # Constrain tree depth to prevent leaf memorization
        min_samples_leaf=8,        # Require at least 8 samples per leaf
        min_samples_split=16,      # Require at least 16 samples to create a split
        max_features="sqrt",       # Feature bagging (decorrelates trees)
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",
    ),
    "xgb": dict(
        n_estimators=300,
        learning_rate=0.08,
        max_depth=5,               # Constrain tree depth
        min_child_weight=4,        # Regularize leaf weight sum
        gamma=1.0,                 # Minimum loss reduction for split
        subsample=0.75,            # Sample 75% rows per tree
        colsample_bytree=0.65,     # Sample 65% features per tree
        reg_alpha=0.5,             # L1 regularization on weights
        reg_lambda=1.5,            # L2 regularization on weights
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    ),
    "lgbm": dict(
        n_estimators=300,
        learning_rate=0.08,
        max_depth=5,               # Constrain depth
        num_leaves=24,             # Limit maximum leaf count
        min_child_samples=15,      # Minimum samples per leaf
        subsample=0.75,
        colsample_bytree=0.65,
        reg_alpha=0.5,
        reg_lambda=1.5,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        class_weight="balanced",
    ),
    "svm": dict(
        C=1.0,
        gamma="scale",
        kernel="rbf",
        probability=True,
        random_state=42,
        class_weight="balanced",
    ),
}


# ---------------------------------------------------------------------------
# Hyper-parameter search grids (for RandomizedSearchCV)
# ---------------------------------------------------------------------------

PARAM_GRIDS: dict[str, dict] = {
    "rf": {
        "n_estimators": [150, 250, 350],
        "max_depth": [5, 7, 9, 12],
        "min_samples_leaf": [4, 8, 16],
        "min_samples_split": [8, 16, 24],
        "max_features": ["sqrt", "log2"],
    },
    "xgb": {
        "n_estimators": [150, 250, 350],
        "learning_rate": [0.03, 0.08, 0.12],
        "max_depth": [4, 5, 6],
        "min_child_weight": [3, 5, 8],
        "subsample": [0.65, 0.75, 0.85],
        "colsample_bytree": [0.55, 0.65, 0.8],
        "reg_alpha": [0.1, 0.5, 1.0],
        "reg_lambda": [1.0, 2.0, 5.0],
    },
    "lgbm": {
        "n_estimators": [150, 250, 350],
        "learning_rate": [0.03, 0.08, 0.12],
        "max_depth": [4, 5, 6],
        "num_leaves": [16, 24, 32],
        "min_child_samples": [10, 15, 25],
        "subsample": [0.65, 0.75, 0.85],
        "colsample_bytree": [0.55, 0.65, 0.8],
        "reg_alpha": [0.1, 0.5, 1.0],
        "reg_lambda": [1.0, 2.0, 5.0],
    },
    "svm": {
        "clf__C": [0.1, 1.0, 5.0, 10.0],
        "clf__gamma": ["scale", "auto", 0.01, 0.05],
    },
}


# ---------------------------------------------------------------------------
# Quick-access list of all model keys
# ---------------------------------------------------------------------------

ALL_MODELS = list(MODEL_NAMES.keys())
