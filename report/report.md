# Phishing URL Detection — Generalisation Study
## Mini-Project Report

**Student:** [Your Name]
**Module:** Applied Machine Learning for Cybersecurity
**Date:** September 2026

---

## Abstract

Phishing URL classifiers trained and tested on the same benchmark dataset routinely achieve accuracy
exceeding 97%, yet they often fail to generalise to URLs collected from different sources. This
project builds a phishing URL classification framework using three families of traditional
machine-learning models — Random Forest, XGBoost/LightGBM, and SVM — and investigates how
performance changes when evaluation switches from an in-distribution random split to a
cross-dataset protocol. A 25-feature lexical feature set is extracted from raw URL strings.
Results are compared across accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix, and
false-positive rate. SHAP analysis and Jensen-Shannon divergence measurements identify the features
responsible for performance degradation. Two practical mitigations — feature selection and data
blending — are evaluated. The project concludes with a discussion of whether high benchmark
accuracy reliably predicts real-world detection capability.

---

## 1. Introduction

### 1.1 Problem Statement

Phishing attacks remain one of the most common social-engineering vectors in cybersecurity. URL
inspection is a critical first line of defence because it operates before the user reaches a
malicious page. Supervised machine-learning classifiers trained on curated datasets (e.g.,
PhishTank + Alexa) achieve near-perfect accuracy on random test splits drawn from the same dataset.
However, **when deployed against URLs from a different campaign, collection period, or source,
performance degrades substantially** — a phenomenon called *dataset shift* or
*covariate shift*.

This project quantifies that gap and analyses its root causes.

### 1.2 Objectives

1. Build a phishing URL classification framework using URL-only lexical features.
2. Train and evaluate Random Forest, XGBoost, LightGBM, and SVM classifiers.
3. Compare in-distribution (random split) performance with cross-dataset performance.
4. Use SHAP and distribution shift analysis to identify which features cause degradation.
5. Test two practical mitigations: feature selection and data blending.

---

## 2. Related Work

*(Fill in: cite 3–5 relevant papers here. Suggested: Mohammad et al. 2012; Sahingoz et al. 2019;
Shirazi et al. 2021 on dataset shift in phishing detection; Srinivasa et al. 2021 ISCX analysis.)*

---

## 3. Methodology

### 3.1 Datasets

| ID | Name | Source | Size | Label |
|---|---|---|---|---|
| **D1** | UCI Phishing Websites | [UCI ML Repository](https://archive.ics.uci.edu/dataset/327/phishing+websites) | ~11,055 rows | Heuristic binary/ternary |
| **D2** | PhiUSIIL Phishing URL | [Kaggle](https://www.kaggle.com/datasets/harisudhan411/phishing-and-legitimate-urls) | ~235,000 rows | Raw URL + binary label |

D1 provides 30 pre-extracted heuristic/behavioural features. D2 contains raw URLs. Both datasets
are mapped to a **common 25-feature lexical space** to enable cross-dataset transfer evaluation.

### 3.2 Feature Engineering

Twenty-five features are extracted from raw URL strings using the `src/features.py` module:

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

All features are derived purely from URL string parsing — no DNS, WHOIS, or content fetching is
required.

### 3.3 Models

| Model | Hyper-parameter Tuning |
|---|---|
| Random Forest | 5-fold CV, RandomizedSearchCV (20 iterations) |
| XGBoost | 5-fold CV, RandomizedSearchCV (20 iterations) |
| LightGBM | 5-fold CV, RandomizedSearchCV (20 iterations) |
| SVM (RBF) | 5-fold CV with StandardScaler pipeline |

### 3.4 Evaluation Protocol

**Experiment 1 — In-distribution:**
- D1: 80/20 stratified random split
- D2: 80/20 stratified random split

**Experiment 2 — Cross-dataset:**
- Train on D1 → Test on D2
- Train on D2 → Test on D1

**Metrics:** Accuracy, Precision, Recall, F1 (binary, macro, weighted), ROC-AUC, PR-AUC,
Confusion Matrix, False-Positive Rate.

---

## 4. Results

> *Replace the placeholder values below with your actual results after running the experiments.*

### 4.1 Experiment 1 — In-distribution Performance

| Model | Dataset | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR |
|---|---|---|---|---|---|---|---|
| Random Forest | D1 | — | — | — | — | — | — |
| XGBoost | D1 | — | — | — | — | — | — |
| LightGBM | D1 | — | — | — | — | — | — |
| SVM (RBF) | D1 | — | — | — | — | — | — |
| Random Forest | D2 | — | — | — | — | — | — |
| XGBoost | D2 | — | — | — | — | — | — |
| LightGBM | D2 | — | — | — | — | — | — |
| SVM (RBF) | D2 | — | — | — | — | — | — |

### 4.2 Experiment 2 — Cross-dataset Performance

| Model | Train → Test | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR |
|---|---|---|---|---|---|---|---|
| Random Forest | D1 → D2 | — | — | — | — | — | — |
| XGBoost | D1 → D2 | — | — | — | — | — | — |
| LightGBM | D1 → D2 | — | — | — | — | — | — |
| SVM (RBF) | D1 → D2 | — | — | — | — | — | — |
| Random Forest | D2 → D1 | — | — | — | — | — | — |
| XGBoost | D2 → D1 | — | — | — | — | — | — |
| LightGBM | D2 → D1 | — | — | — | — | — | — |
| SVM (RBF) | D2 → D1 | — | — | — | — | — | — |

### 4.3 Generalisation Gap (Δ-F1)

| Model | D1 in-dist F1 | D1→D2 F1 | **Δ** | D2 in-dist F1 | D2→D1 F1 | **Δ** |
|---|---|---|---|---|---|---|
| Random Forest | — | — | — | — | — | — |
| XGBoost | — | — | — | — | — | — |
| LightGBM | — | — | — | — | — | — |
| SVM (RBF) | — | — | — | — | — | — |

---

## 5. Analysis

### 5.1 Feature Importance (SHAP)

*(Insert `results/figures/shap_rf_beeswarm.png` and discuss top features.)*

Key findings:
- The most important features for Random Forest are: [to be filled]
- XGBoost ranks differently, with: [to be filled]
- Features that agree across both models are the most reliable signals.

### 5.2 Distribution Shift

*(Insert `results/figures/distribution_shift_heatmap.png` and discuss.)*

Features with highest JS divergence between D1 and D2:
- [to be filled from `results/metrics/distribution_shift.csv`]

Features most likely causing generalisation failure (high SHAP + high JS):
- [to be filled from `results/figures/shap_vs_shift_scatter.png`]

### 5.3 Error Analysis

*(Insert `results/figures/error_analysis_fn_vs_tp.png` and discuss patterns in missed phishing.)*

---

## 6. Mitigation Results

### 6.1 Feature Selection

Using only features with JS < 0.3 (stable across datasets):

| Model | All Features F1 | Stable Features F1 | Δ |
|---|---|---|---|
| Random Forest | — | — | — |
| XGBoost | — | — | — |

### 6.2 Data Blending

*(Insert `results/figures/blending_mitigation_curve.png` and discuss optimal blend ratio.)*

---

## 7. Discussion

### 7.1 Does high benchmark accuracy predict real-world performance?

*(Write 2–3 paragraphs analysing the generalisation gap observed and what drives it.)*

### 7.2 Which model generalises best?

*(Rank models by cross-dataset F1 and explain why ensemble methods may outperform SVM.)*

### 7.3 Most effective mitigation strategy

*(Compare feature selection vs. data blending and recommend a practical approach.)*

### 7.4 Limitations

- Lexical features only — no content or behavioural signals.
- Both datasets are static snapshots; phishing campaigns evolve rapidly.
- D1 is small (~11K rows); conclusions on D1 may have high variance.

---

## 8. Conclusion

*(Write a 1-paragraph conclusion summarising key findings and answering the research question.)*

---

## References

1. Mohammad, R. M., Thabtah, F., & McCluskey, L. (2012). Intelligent rule-based phishing websites
   classification. *IET Information Security*, 8(3), 153–160.
2. UCI ML Repository – Phishing Websites Dataset.
   https://archive.ics.uci.edu/dataset/327/phishing+websites
3. PhiUSIIL Phishing URL Dataset. https://www.kaggle.com/datasets/harisudhan411/phishing-and-legitimate-urls
4. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions.
   *Advances in Neural Information Processing Systems*, 30.
5. *(Add further references here.)*

---

## Appendix — Figures

*(After running experiments, embed or reference the following figures:)*

- `results/figures/eda_class_balance.png`
- `results/figures/eda_url_lengths.png`
- `results/figures/feature_mutual_info.png`
- `results/figures/cm_exp1_D1-UCI_rf.png` *(and other confusion matrices)*
- `results/figures/roc_pr_D1-UCI_exp1.png`
- `results/figures/shap_rf_beeswarm.png`
- `results/figures/shap_xgb_beeswarm.png`
- `results/figures/shap_vs_shift_scatter.png`
- `results/figures/distribution_shift_heatmap.png`
- `results/figures/distribution_shift_violins.png`
- `results/figures/generalisation_gap_f1.png`
- `results/figures/blending_mitigation_curve.png`
