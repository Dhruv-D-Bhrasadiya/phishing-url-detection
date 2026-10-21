# Phishing URL Detection — Generalisation Study

A mini-project that investigates how well phishing URL classifiers trained on one benchmark dataset
generalise to URLs collected from a different source.

---

## Project Structure

```
aml (Cybersecurity)/
├── data/
│   ├── raw/                    # Place downloaded CSVs here
│   └── processed/              # Feature-engineered parquet files (auto-generated)
├── src/
│   ├── features.py             # 25-feature lexical URL extractor
│   ├── data_loader.py          # Dataset loading & preprocessing
│   ├── models.py               # Model factory + hyper-parameter grids
│   ├── evaluation.py           # Metrics, plots, SHAP, distribution shift
│   └── experiments.py          # CLI entry point — runs all experiments
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_experiments.ipynb    # Reproduces all experiments interactively
│   └── 04_analysis.ipynb       # SHAP, distribution shift, generalisation gap
├── results/
│   ├── figures/                # All plots saved as PNG
│   └── metrics/                # JSON/CSV result tables
├── report/
│   └── report.md               # Final written report
├── requirements.txt
└── README.md
```

---

## Datasets

### D1 — UCI Phishing Websites (30 pre-extracted features)

1. Download from: https://archive.ics.uci.edu/dataset/327/phishing+websites
2. Save the CSV as: `data/raw/dataset_phishing_uci.csv`

### D2 — PhiUSIIL Phishing URL Dataset (raw URLs)

1. Download from: https://www.kaggle.com/datasets/harisudhan411/phishing-and-legitimate-urls
2. Save the CSV as: `data/raw/phiusiil_phishing_url.csv`

> **Note:** If you have a differently named or structured CSV, update the paths in the
> `--d1` and `--d2` arguments, or adjust `data_loader.py` accordingly.

---

## Setup

```powershell
# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

---

## How to Run (Master Launcher)

The easiest way to run any part of the project is using the interactive launcher:

```powershell
.\run.bat
```
This presents a menu to:
1. **Launch Web UI** (`http://127.0.0.1:5000`)
2. **Run Quick Demo** (30-second smoke-test)
3. **Run Full Experiments** (Train RF, XGBoost, LightGBM, compute metrics & plots)
4. **Open Jupyter Notebooks**
5. **Run Unit Tests** (`pytest tests/`)

Or launch the Web UI directly with:
```powershell
.\run_ui.bat
```

---

## Running Experiments

### Quick smoke-test (no data required)

```powershell
python -m src.experiments --quick-check
```

### Full experiment run

```powershell
python -m src.experiments `
    --d1 data/raw/dataset_phishing_uci.csv `
    --d2 data/raw/phiusiil_phishing_url.csv
```

### Speed-optimised run (sample + no tuning)

```powershell
python -m src.experiments `
    --d1 data/raw/dataset_phishing_uci.csv `
    --d2 data/raw/phiusiil_phishing_url.csv `
    --d1-sample 5000 --d2-sample 10000 `
    --no-tune --skip-bonus
```

### Run specific models only

```powershell
python -m src.experiments `
    --d1 data/raw/dataset_phishing_uci.csv `
    --d2 data/raw/phiusiil_phishing_url.csv `
    --models rf xgb
```

---

## Features (25 Lexical)

Extracted entirely from the URL string — no network calls required.

| Category | Features |
|---|---|
| Length | `url_length`, `domain_length`, `path_length`, `query_length` |
| Counts | `n_dots`, `n_hyphens`, `n_underscores`, `n_slashes`, `n_at`, `n_digits`, `n_special_chars` |
| Structural | `has_ip`, `has_port`, `is_https`, `subdomain_level`, `path_depth` |
| Entropy | `url_entropy`, `domain_entropy` |
| Suspicious tokens | `has_suspicious_word`, `n_suspicious_words` |
| Shortening/redirect | `uses_shortener`, `has_redirect` |
| TLD | `tld_is_common` |
| Brand impersonation | `brand_in_subdomain`, `brand_in_path` |

---

## Models

| Key | Model | Library |
|---|---|---|
| `rf` | Random Forest | scikit-learn |
| `xgb` | XGBoost | xgboost |
| `lgbm` | LightGBM | lightgbm |
| `svm` | SVM (RBF) | scikit-learn |

All models are tuned with `RandomizedSearchCV` (5-fold stratified CV) unless `--no-tune` is set.

---

## Evaluation

### Experiment 1 — In-distribution (random 80/20 split)
- D1 only: train 80%, test 20%
- D2 only: train 80%, test 20%

### Experiment 2 — Cross-dataset (generalisation)
- Train D1 → Test D2
- Train D2 → Test D1

### Metrics
Accuracy, Precision, Recall, F1, F1-macro, F1-weighted, ROC-AUC, PR-AUC, FPR, Confusion Matrix

---

## Outputs

After running:

```
results/
├── figures/
│   ├── cm_*.png                        # Confusion matrices
│   ├── roc_pr_*.png                    # ROC & PR curves
│   ├── feat_imp_*.png                  # Feature importance bar charts
│   ├── shap_rf.png / shap_xgb.png     # SHAP beeswarm plots
│   ├── distribution_shift_violins.png  # Feature distribution comparison
│   ├── distribution_shift_heatmap.png  # JS divergence per feature
│   ├── generalisation_gap_f1.png       # Gap between random-split and cross-dataset F1
│   └── metrics_heatmap.png             # Full metrics overview
└── metrics/
    ├── all_metrics.json                # All experiment metrics
    └── distribution_shift.csv          # KL/JS divergence per feature
```

---

## Notebooks

Open Jupyter and run notebooks in order:

```powershell
jupyter notebook
```

1. `01_eda.ipynb` — data exploration, class balance, URL length distributions
2. `02_feature_engineering.ipynb` — feature extraction walk-through, feature correlation
3. `03_experiments.ipynb` — interactive version of all experiments with inline plots
4. `04_analysis.ipynb` — SHAP analysis, distribution shift deep-dive, generalisation gap

---

## References

- Phishing Dataset (UCI): https://archive.ics.uci.edu/dataset/327/phishing+websites
- PhiUSIIL Dataset: https://www.kaggle.com/datasets/harisudhan411/phishing-and-legitimate-urls
- ISCX-URL2016: https://www.unb.ca/cic/datasets/url-2016.html
- Mohammad et al. (2012) "Intelligent Rule-based Phishing Websites Classification"

---

## Ray Tune: Automated Hyperparameter Experiments

The optional Ray Tune runner searches a bounded hyperparameter space for Random Forest, XGBoost, and LightGBM. It distributes trials across available CPU resources, ranks configurations by validation F1, then refits the best configuration on the training partition and evaluates it once on a held-out test partition. The test set is not used for hyperparameter selection.

Install dependencies (Ray is included in requirements.txt):

```bash
pip install -r requirements.txt
```

Run a reproducible search on the PhiUSIIL CSV:

```bash
python -m src.ray_experiments \
  --d2 data/raw/phiusiil_phishing_url.csv \
  --sample-size 20000 \
  --num-samples 12 \
  --models rf xgb lgbm \
  --seed 42
```

Arguments:
- `--d2`: input CSV path (defaults to `data/raw/phiusiil_phishing_url.csv`).
- `--sample-size`: number of rows sampled for a faster run; set `0` to use all rows.
- `--num-samples`: total number of configurations across the selected model families.
- `--models`: one or more of `rf`, `xgb`, `lgbm`.
- `--cpus-per-trial`: CPU resources reserved for each Ray trial (default `2`).
- `--seed`: random seed for reproducibility.

Ray results are written to `results/ray/`:
- `metrics/ray_trials.csv`: validation metrics and configuration for every successful trial.
- `metrics/best_result.json`: best configuration and held-out test metrics.
- `figures/best_confusion_matrix.png`: held-out test confusion matrix.
- `reports/ray_experiment_report.md`: metrics and classification report.
- `ray_storage/`: Ray Tune trial logs and experiment state.

The default search is bounded to 12 configurations and samples up to 20,000 rows to keep the first run manageable. Increase `--num-samples` or set `--sample-size 0` when more compute is available. The Ray runner is separate from `src.experiments`; existing experiments continue to work as before.
