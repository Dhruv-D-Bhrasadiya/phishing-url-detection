"""
data_loader.py
--------------
Loads and preprocesses the two phishing URL datasets:

  D1 -- UCI Phishing Websites dataset (30 pre-extracted heuristic features)
       CSV downloaded from UCI ML Repository.
       Label column: 'Result'  {-1: phishing, 1: legitimate}

  D2 -- PhiUSIIL Phishing URL Dataset (raw URLs + label)
       CSV from Kaggle.
       Expected columns: 'URL', 'label'  {1: phishing, 0: legitimate}

Both are harmonised to:
  - Feature columns: the 25 lexical features (from src/features.py)
  - Label column: 'label'  {1: phishing, 0: legitimate}

Usage
-----
    from src.data_loader import load_d1, load_d2, load_both

    X1, y1 = load_d1("data/raw/dataset_phishing_uci.csv")
    X2, y2 = load_d2("data/raw/phiusiil_phishing_url.csv")
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import extract_features_batch, FEATURE_NAMES

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {p}\n"
            "Please download the dataset and place it at this path.\n"
            "See README.md for download instructions."
        )
    return p


def _balance_info(y: pd.Series, name: str) -> None:
    counts = y.value_counts()
    total = len(y)
    print(f"[{name}] Total: {total:,}  |  "
          f"Phishing: {counts.get(1, 0):,} ({100*counts.get(1,0)/total:.1f}%)  |  "
          f"Legit: {counts.get(0, 0):,} ({100*counts.get(0,0)/total:.1f}%)")


# ---------------------------------------------------------------------------
# D1 -- UCI Phishing Websites (pre-extracted features)
# ---------------------------------------------------------------------------

# The UCI 30-feature dataset uses heuristic binary/ternary encodings.
# We extract our 25 lexical features instead by regenerating them from
# whatever URL column is present. If no URL column exists we fall back to
# using a synthetic URL reconstruction is not possible, so we keep the
# UCI features mapped to our common set where overlap exists.
#
# For maximum reproducibility the safest approach is:
#   - If a 'url' column is present in the CSV -> extract our 25 lexical features.
#   - Otherwise -> map UCI columns to our feature names where possible,
#     and drop features that have no equivalent.

UCI_LABEL_MAP = {-1: 1, 1: 0}   # UCI: -1=phishing,1=legit -> our: 1=phishing,0=legit

# Mapping from UCI column name -> our feature name (where 1-to-1 is sensible)
UCI_TO_LEXICAL = {
    "URL_Length":        "url_length",         # UCI uses 1/0/-1; we store int count
    "having_IP_Address": "has_ip",
    "SSLfinal_State":    "is_https",
    "having_At_Sign":    "n_at",
    "double_slash_redirecting": "has_redirect",
    "Prefix_Suffix":     "n_hyphens",          # heuristic proxy
    "sub_Domain":        "subdomain_level",
}


def load_d1(
    csv_path: str | Path,
    use_lexical: bool = True,
    n_jobs: int = -1,
    sample: int | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load the UCI Phishing Websites dataset.

    Parameters
    ----------
    csv_path : path-like
        Path to the downloaded UCI CSV file.
    use_lexical : bool
        If True and a 'url'/'URL' column exists, extract our 25 lexical
        features from it. If False (or no URL column), use UCI's own
        pre-extracted features (subset mapped to our names).
    n_jobs : int
        Parallel workers for feature extraction.
    sample : int or None
        If given, randomly sample this many rows (reproducible).

    Returns
    -------
    X : pd.DataFrame  -- feature matrix (25 columns)
    y : pd.Series     -- labels (1=phishing, 0=legitimate)
    """
    p = _resolve(csv_path)
    df = pd.read_csv(p)

    # Normalise column names
    df.columns = [c.strip() for c in df.columns]

    # Detect label column
    label_col = None
    for candidate in ["Result", "result", "label", "Label", "class", "Class"]:
        if candidate in df.columns:
            label_col = candidate
            break
    if label_col is None:
        raise ValueError(f"Cannot find label column in {p}. Columns: {list(df.columns)}")

    # Map labels -> {1: phishing, 0: legit}
    raw_labels = df[label_col]
    if raw_labels.isin([-1, 1]).all():
        y = raw_labels.map(UCI_LABEL_MAP).astype(int)
    else:
        y = raw_labels.astype(int)

    # Detect URL column
    url_col = next((c for c in df.columns if c.lower() == "url"), None)

    if use_lexical and url_col is not None:
        urls = df[url_col]
        if sample:
            idx = urls.sample(min(sample, len(urls)), random_state=42).index
            urls, y = urls.loc[idx], y.loc[idx]
        print(f"[D1] Extracting lexical features from URL column '{url_col}'...")
        X = extract_features_batch(urls, n_jobs=n_jobs)
    else:
        # Fall back to UCI pre-extracted columns
        feature_cols = [c for c in df.columns if c != label_col and c != url_col]
        X_raw = df[feature_cols].copy()
        if sample:
            idx = X_raw.sample(min(sample, len(X_raw)), random_state=42).index
            X_raw, y = X_raw.loc[idx], y.loc[idx]
        # Rename overlapping columns to our names
        rename_map = {k: v for k, v in UCI_TO_LEXICAL.items() if k in X_raw.columns}
        X_raw = X_raw.rename(columns=rename_map)
        # Keep only columns that exist in our feature set, pad missing with 0
        X = pd.DataFrame(0, index=X_raw.index, columns=FEATURE_NAMES)
        for col in X_raw.columns:
            if col in FEATURE_NAMES:
                X[col] = X_raw[col]
        # UCI length columns are ordinal {-1,0,1}; convert to approximate counts
        if "url_length" in X.columns:
            X["url_length"] = X["url_length"].map({-1: 90, 0: 60, 1: 30}).fillna(60)

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    _balance_info(y, "D1-UCI")
    return X, y


# ---------------------------------------------------------------------------
# D2 -- PhiUSIIL / Kaggle raw-URL dataset
# ---------------------------------------------------------------------------

def load_d2(
    csv_path: str | Path,
    n_jobs: int = -1,
    sample: int | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load the PhiUSIIL (or similar) raw-URL phishing dataset.

    Expected columns
    ----------------
    * 'URL' or 'url'  -- raw URL string
    * 'label' or 'Label'  -- {1: phishing, 0: legitimate}

    Parameters
    ----------
    csv_path : path-like
    n_jobs : int
    sample : int or None

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    """
    p = _resolve(csv_path)
    df = pd.read_csv(p, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    # URL column
    url_col = next(
        (c for c in df.columns if c.lower() in ("url", "urls", "url_string")), None
    )
    if url_col is None:
        raise ValueError(f"Cannot find URL column in {p}. Columns: {list(df.columns)}")

    # Label column
    label_col = next(
        (c for c in df.columns if c.lower() in ("label", "class", "result", "status", "type")),
        None,
    )
    if label_col is None:
        raise ValueError(f"Cannot find label column in {p}. Columns: {list(df.columns)}")

    df = df[[url_col, label_col]].dropna()
    df[label_col] = df[label_col].astype(int)

    # Normalise labels: ensure 1=phishing, 0=legit
    unique_labels = set(df[label_col].unique())
    if unique_labels == {-1, 1}:
        df[label_col] = df[label_col].map({-1: 1, 1: 0})

    if sample:
        df = df.sample(min(sample, len(df)), random_state=42)

    urls = df[url_col].reset_index(drop=True)
    y = df[label_col].reset_index(drop=True)

    print(f"[D2] Extracting lexical features from {len(urls):,} URLs...")
    X = extract_features_batch(urls, n_jobs=n_jobs)
    X = X.reset_index(drop=True)

    _balance_info(y, "D2-PhiUSIIL")
    return X, y


# ---------------------------------------------------------------------------
# Convenience loader for both datasets
# ---------------------------------------------------------------------------

def load_both(
    d1_path: str | Path,
    d2_path: str | Path,
    d1_use_lexical: bool = True,
    n_jobs: int = -1,
    d1_sample: int | None = None,
    d2_sample: int | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load D1 and D2 and return feature matrices and label vectors.

    Returns
    -------
    X1, y1, X2, y2
    """
    X1, y1 = load_d1(d1_path, use_lexical=d1_use_lexical, n_jobs=n_jobs, sample=d1_sample)
    X2, y2 = load_d2(d2_path, n_jobs=n_jobs, sample=d2_sample)
    return X1, y1, X2, y2


# ---------------------------------------------------------------------------
# Synthetic dataset generator (for smoke-testing without real data)
# ---------------------------------------------------------------------------

_PHISHING_URLS = [
    "http://paypal-login.ru/verify/account?id=123",
    "http://192.168.1.1/bank/login.php",
    "http://secure-amazon.tk/update/credentials",
    "http://bit.ly/3xFakePhish",
    "http://ebay-signin.support/user/verify",
]

_LEGIT_URLS = [
    "https://www.google.com/search?q=news",
    "https://en.wikipedia.org/wiki/Phishing",
    "https://github.com/openai/openai-python",
    "https://www.amazon.com/dp/B08N5WRWNW",
    "https://stackoverflow.com/questions/12345",
]


def generate_synthetic_dataset(
    n_phishing: int = 500,
    n_legit: int = 500,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Generate a small synthetic dataset for smoke-testing.

    Phishing URLs are randomly mutated from templates; legit URLs from
    common domains. No ground truth ML value -- for pipeline testing only.
    """
    rng = np.random.default_rng(random_state)

    def _mutate_phishing(url: str, rng: np.random.Generator) -> str:
        extra = "".join(rng.choice(list("abcdefghijklmnop0123456789"), size=8))
        return url + "/" + extra

    def _mutate_legit(url: str, rng: np.random.Generator) -> str:
        path = "/".join(
            "".join(rng.choice(list("abcdefghijklmnop"), size=5))
            for _ in range(rng.integers(1, 4))
        )
        return url + "/" + path

    phish_urls = [
        _mutate_phishing(_PHISHING_URLS[i % len(_PHISHING_URLS)], rng)
        for i in range(n_phishing)
    ]
    legit_urls = [
        _mutate_legit(_LEGIT_URLS[i % len(_LEGIT_URLS)], rng)
        for i in range(n_legit)
    ]

    all_urls = phish_urls + legit_urls
    labels = [1] * n_phishing + [0] * n_legit

    X = extract_features_batch(pd.Series(all_urls), n_jobs=1)
    y = pd.Series(labels, name="label")
    return X, y
