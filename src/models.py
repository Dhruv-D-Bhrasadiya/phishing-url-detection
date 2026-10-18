"""
Regularised model factory for phishing URL experiments.

Defaults deliberately favour generalisation over memorising the training set.
Tune only on the training partition; never select settings from the held-out test set.
"""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

MODEL_NAMES = {
    "rf": "Random Forest",
    "xgb": "XGBoost",
    "lgbm": "LightGBM",
    "svm": "SVM (RBF)",
}

_DEFAULTS = {
    "rf": dict(
        n_estimators=250, max_depth=6, min_samples_leaf=12,
        min_samples_split=24, max_features=0.7, bootstrap=True,
        max_samples=0.8, n_jobs=-1, random_state=42,
        class_weight="balanced_subsample",
    ),
    "xgb": dict(
        n_estimators=250, learning_rate=0.035, max_depth=3,
        min_child_weight=8, gamma=1.5, subsample=0.75,
        colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=5.0,
        objective="binary:logistic", eval_metric="logloss",
        random_state=42, n_jobs=-1, verbosity=0,
    ),
    "lgbm": dict(
        n_estimators=250, learning_rate=0.035, max_depth=4,
        num_leaves=12, min_child_samples=30, subsample=0.75,
        subsample_freq=1, colsample_bytree=0.7, reg_alpha=1.0,
        reg_lambda=5.0, random_state=42, n_jobs=-1,
        verbosity=-1, class_weight="balanced",
    ),
    "svm": dict(
        C=0.5, gamma="scale", kernel="rbf", probability=True,
        random_state=42, class_weight="balanced",
    ),
}

def get_model(key: str, **kwargs):
    """Return an unfitted estimator with conservative regularisation."""
    if key not in _DEFAULTS:
        raise ValueError(f"Unknown model key {key!r}; choose from {list(MODEL_NAMES)}")
    params = {**_DEFAULTS[key], **kwargs}
    if key == "rf":
        return RandomForestClassifier(**params)
    if key == "xgb":
        return XGBClassifier(**params)
    if key == "lgbm":
        return LGBMClassifier(**params)
    if key == "svm":
        return Pipeline([("scaler", StandardScaler()), ("clf", SVC(**params))])
    raise ValueError(key)

PARAM_GRIDS = {
    "rf": {
        "n_estimators": [150, 250],
        "max_depth": [4, 6, 8],
        "min_samples_leaf": [8, 12, 20],
        "min_samples_split": [16, 24, 40],
        "max_features": ["sqrt", 0.7],
    },
    "xgb": {
        "n_estimators": [100, 180, 250],
        "learning_rate": [0.02, 0.035, 0.05],
        "max_depth": [2, 3, 4],
        "min_child_weight": [5, 8, 12],
        "subsample": [0.65, 0.75, 0.85],
        "colsample_bytree": [0.55, 0.7, 0.85],
        "reg_alpha": [0.5, 1.0, 2.0],
        "reg_lambda": [3.0, 5.0, 10.0],
        "gamma": [0.5, 1.5, 3.0],
    },
    "lgbm": {
        "n_estimators": [100, 180, 250],
        "learning_rate": [0.02, 0.035, 0.05],
        "max_depth": [3, 4, 5],
        "num_leaves": [7, 12, 16],
        "min_child_samples": [20, 30, 50],
        "subsample": [0.65, 0.75, 0.85],
        "colsample_bytree": [0.55, 0.7, 0.85],
        "reg_alpha": [0.5, 1.0, 2.0],
        "reg_lambda": [3.0, 5.0, 10.0],
    },
    "svm": {
        "clf__C": [0.05, 0.1, 0.5, 1.0],
        "clf__gamma": ["scale", "auto", 0.001, 0.01],
    },
}
ALL_MODELS = list(MODEL_NAMES)
