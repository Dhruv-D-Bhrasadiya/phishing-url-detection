"""
src/app.py
----------
Production Flask Web Application & REST API for Phishing URL Detection & AML Research.
Provides:
  - Real-time lexical feature extraction & ML inference (RF, XGBoost, LightGBM)
  - Threat indicator breakdown & explainability
  - Cross-dataset generalization metrics & evaluation results
  - OOD Distribution shift analysis (JS Divergence & Wasserstein distance)
  - High-resolution research figure gallery & visualization inspection
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List

# Ensure root directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template, send_from_directory

from src.features import extract_features, FEATURE_NAMES
from src.models import get_model

app = Flask(__name__, template_folder="templates", static_folder="static")

MODELS_DIR = ROOT_DIR / "results" / "models"
FIGURES_DIR = ROOT_DIR / "results" / "figures"
METRICS_DIR = ROOT_DIR / "results" / "metrics"
DATA_DIR = ROOT_DIR / "data" / "raw"

MODELS = {}

# Sample URLs for one-click testing in UI
SAMPLE_URLS = [
    {
        "url": "http://paypal-security-update.account-verification-login.com/webscr",
        "category": "High Risk Phishing",
        "type": "phishing",
        "notes": "Brand impersonation in subdomain + credential theft keywords"
    },
    {
        "url": "http://192.168.1.105:8080/bank/login-verify.php?token=938210",
        "category": "High Risk Phishing",
        "type": "phishing",
        "notes": "Raw IP address host + non-standard port + suspicious terms"
    },
    {
        "url": "http://bit.ly/free-netflix-giftcard-claim-now2024",
        "category": "Suspicious / Shortener",
        "type": "phishing",
        "notes": "URL shortener hiding destination + urgent promo lure"
    },
    {
        "url": "https://www.google.com/search?q=cybersecurity+machine+learning",
        "category": "Legitimate Safe",
        "type": "legitimate",
        "notes": "Official HTTPS domain, standard search query structure"
    },
    {
        "url": "https://github.com/scikit-learn/scikit-learn/releases",
        "category": "Legitimate Safe",
        "type": "legitimate",
        "notes": "Renowned repository platform, clean domain structure"
    },
    {
        "url": "https://en.wikipedia.org/wiki/Phishing",
        "category": "Legitimate Safe",
        "type": "legitimate",
        "notes": "High reputation educational encyclopedia domain"
    }
]

FEATURE_DESCRIPTIONS = {
    "url_length": "Total length of the URL in characters",
    "domain_length": "Length of the fully-qualified domain name",
    "path_length": "Length of the URL path component",
    "query_length": "Length of query parameters string",
    "n_dots": "Number of dot (.) delimiters in URL",
    "n_hyphens": "Number of hyphens (-) in URL (often used for token deception)",
    "n_underscores": "Number of underscores (_) in URL",
    "n_slashes": "Number of forward slashes (/) indicating directory depth",
    "n_at": "Presence of '@' character (used to disguise actual destination)",
    "n_digits": "Total count of numerical digits in URL string",
    "n_special_chars": "Count of non-standard punctuation / special characters",
    "has_ip": "Flag indicating if host is a raw IPv4 address",
    "has_port": "Flag indicating if non-standard port is specified",
    "is_https": "Security protocol flag (1 if HTTPS, 0 if HTTP/none)",
    "subdomain_level": "Number of subdomain hierarchies (excessive levels suspicious)",
    "path_depth": "Folder depth of the URL path",
    "url_entropy": "Shannon information entropy across full URL (randomness indicator)",
    "domain_entropy": "Shannon information entropy of domain (detects DGA algorithms)",
    "has_suspicious_word": "Presence of high-risk security/credential terms",
    "n_suspicious_words": "Total count of credential-harvesting tokens detected",
    "uses_shortener": "Flag indicating URL shortening service (bit.ly, t.co, etc.)",
    "has_redirect": "Presence of double slash (//) redirect redirection pattern",
    "tld_is_common": "Flag indicating common trusted top-level domain (.com, .org, etc.)",
    "brand_in_subdomain": "Flag indicating known brand name spoofed in subdomain",
    "brand_in_path": "Flag indicating known brand name injected in path"
}

def ensure_trained_models():
    """Load cached models from results/models/ or train them fast."""
    global MODELS
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    needed = ["rf", "xgb", "lgbm"]
    loaded = {}
    
    for key in needed:
        model_path = MODELS_DIR / f"{key}_model.joblib"
        if model_path.exists():
            try:
                loaded[key] = joblib.load(model_path)
                print(f"[Model Loader] Loaded {key} from {model_path}")
            except Exception as e:
                print(f"[Model Loader] Failed loading {key}: {e}")
                
    if len(loaded) == len(needed):
        MODELS = loaded
        return

    print("[Model Loader] Cached models not found or incomplete. Training robust multi-source models for UI inference...")
    d1_file = DATA_DIR / "dataset_phishing_uci.csv"
    d2_file = DATA_DIR / "phiusiil_phishing_url.csv"

    if d1_file.exists() and d2_file.exists():
        try:
            from src.data_loader import load_d1, load_d2
            print(f"[Model Loader] Loading multi-distribution blend (2,500 D1 + 2,500 D2)...")
            X1, y1 = load_d1(d1_file, sample=2500)
            X2, y2 = load_d2(d2_file, sample=2500)
            X = pd.concat([X1, X2], ignore_index=True)
            y = pd.concat([y1, y2], ignore_index=True)

            for key in needed:
                clf = get_model(key)
                clf.fit(X, y)
                joblib.dump(clf, MODELS_DIR / f"{key}_model.joblib")
                loaded[key] = clf
                print(f"[Model Loader] Trained and saved robust {key} model successfully.")
            MODELS = loaded
            return
        except Exception as e:
            print(f"[Model Loader] Multi-distribution training error: {e}")
    elif d1_file.exists():
        try:
            from src.data_loader import load_d1
            X, y = load_d1(d1_file, sample=4000)
            for key in needed:
                clf = get_model(key)
                clf.fit(X, y)
                joblib.dump(clf, MODELS_DIR / f"{key}_model.joblib")
                loaded[key] = clf
            MODELS = loaded
            return
        except Exception as e:
            print(f"[Model Loader] D1 training error: {e}")

    # Fallback synthetic training if files aren't ready
    print("[Model Loader] Generating bootstrap model with synthetic vectors...")
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    
    # 200 synthetic rows with clear lexical boundary
    rng = np.random.default_rng(42)
    n_samples = 200
    X_syn = pd.DataFrame(rng.uniform(0, 1, size=(n_samples, len(FEATURE_NAMES))), columns=FEATURE_NAMES)
    # Phishing indicators push label towards 1
    y_syn = (X_syn["url_length"] * 2 + X_syn["has_ip"] * 3 + X_syn["has_suspicious_word"] * 4 + X_syn["brand_in_subdomain"] * 3 > 3.5).astype(int)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_syn, y_syn)
    xgb = XGBClassifier(n_estimators=100, random_state=42).fit(X_syn, y_syn)
    lgbm = LGBMClassifier(n_estimators=100, random_state=42).fit(X_syn, y_syn)
    
    joblib.dump(rf, MODELS_DIR / "rf_model.joblib")
    joblib.dump(xgb, MODELS_DIR / "xgb_model.joblib")
    joblib.dump(lgbm, MODELS_DIR / "lgbm_model.joblib")
    
    MODELS = {"rf": rf, "xgb": xgb, "lgbm": lgbm}
    print("[Model Loader] Bootstrap models successfully created.")

# Initialize models on startup
ensure_trained_models()

@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.route("/")
def index():
    """Serve single-page cyber defense dashboard."""
    return render_template("index.html")

@app.route("/api/samples", methods=["GET"])
def get_samples():
    """Return pre-configured sample URLs."""
    return jsonify({"samples": SAMPLE_URLS})

@app.route("/api/scan", methods=["POST"])
def scan_url():
    """Extract features, run inference, compute risk indicators."""
    start_time = time.perf_counter()
    data = request.get_json() or {}
    raw_url = str(data.get("url", ""))
    url = raw_url.strip().strip('\'"<>` \t\r\n')
    model_key = data.get("model", "rf").lower()

    if not url:
        return jsonify({"error": "Please enter a valid URL to scan"}), 400

    if model_key not in MODELS:
        model_key = "rf"
        
    clf = MODELS.get(model_key)
    if clf is None:
        return jsonify({"error": "ML model engine not ready"}), 500

    # 1. Extract 25 lexical features
    try:
        features_dict = extract_features(url)
    except Exception as e:
        return jsonify({"error": f"Lexical extraction failed: {str(e)}"}), 400

    features_df = pd.DataFrame([features_dict], columns=FEATURE_NAMES)

    # 2. Predict with Model
    try:
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(features_df)[0]
            # Class 1 is phishing
            phish_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
        else:
            pred = clf.predict(features_df)[0]
            phish_prob = 1.0 if pred == 1 else 0.0
            
        prediction = 1 if phish_prob >= 0.5 else 0
    except Exception as e:
        return jsonify({"error": f"Inference execution failed: {str(e)}"}), 500

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
    risk_score = round(phish_prob * 100, 1)

    if risk_score >= 70:
        verdict = "MALICIOUS PHISHING"
        status_code = "danger"
        status_desc = "High probability of deceptive credential-harvesting or malicious attack."
    elif risk_score >= 35:
        verdict = "SUSPICIOUS"
        status_code = "warning"
        status_desc = "Moderate anomaly detected. URL contains suspicious lexical markers."
    else:
        verdict = "LEGITIMATE / SAFE"
        status_code = "safe"
        status_desc = "Standard lexical structure. No high-risk deceptive patterns detected."

    # 3. Rule-based cybersecurity Threat Indicators
    indicators = []
    if features_dict.get("has_ip") == 1:
        indicators.append({"level": "danger", "title": "Raw IP Address Host", "desc": "URL uses an IP address instead of a standard domain, heavily associated with malicious campaigns."})
    if features_dict.get("brand_in_subdomain") == 1:
        indicators.append({"level": "danger", "title": "Brand Impersonation in Subdomain", "desc": "Target brand name (e.g. PayPal, Google) appears in subdomain to trick users."})
    if features_dict.get("has_suspicious_word") == 1:
        indicators.append({"level": "warning", "title": f"Security Keywords Detected ({features_dict.get('n_suspicious_words')})", "desc": "Contains sensitive target tokens such as 'login', 'verify', 'account', or 'bank'."})
    if features_dict.get("uses_shortener") == 1:
        indicators.append({"level": "warning", "title": "URL Shortener Service", "desc": "Destination is obscured behind a URL redirection shortener service."})
    if features_dict.get("has_redirect") == 1:
        indicators.append({"level": "warning", "title": "Double Slash Redirector", "desc": "Contains '//' sequence in path to trigger destination redirection."})
    if features_dict.get("subdomain_level", 0) >= 3:
        indicators.append({"level": "warning", "title": "Excessive Subdomain Depth", "desc": f"{features_dict.get('subdomain_level')} subdomain levels detected, commonly used in phishing kits."})
    if features_dict.get("url_length", 0) > 75:
        indicators.append({"level": "info", "title": "Abnormally Long URL", "desc": f"URL length is {features_dict.get('url_length')} characters, often used to conceal real query intent."})
    if features_dict.get("is_https") == 0:
        indicators.append({"level": "info", "title": "Unencrypted HTTP Protocol", "desc": "Site does not enforce HTTPS encryption."})

    if not indicators and verdict == "LEGITIMATE / SAFE":
        indicators.append({"level": "safe", "title": "Clean Lexical Structure", "desc": "No spoofed brand tokens, standard domain entropy, and legitimate URL hierarchy."})

    # Prepare detailed feature list with metadata
    detailed_features = []
    for name in FEATURE_NAMES:
        val = features_dict[name]
        is_risky = False
        if name in ["has_ip", "brand_in_subdomain", "brand_in_path", "has_redirect", "uses_shortener"] and val == 1:
            is_risky = True
        elif name == "has_suspicious_word" and val == 1:
            is_risky = True
        elif name == "is_https" and val == 0:
            is_risky = True
        elif name == "url_length" and val > 80:
            is_risky = True
        elif name == "domain_entropy" and val > 3.8:
            is_risky = True
            
        detailed_features.append({
            "name": name,
            "value": round(val, 4) if isinstance(val, float) else val,
            "description": FEATURE_DESCRIPTIONS.get(name, ""),
            "risky": is_risky
        })

    return jsonify({
        "url": url,
        "model": model_key,
        "prediction": prediction,
        "verdict": verdict,
        "status_code": status_code,
        "status_desc": status_desc,
        "risk_score": risk_score,
        "latency_ms": latency_ms,
        "indicators": indicators,
        "features": features_dict,
        "detailed_features": detailed_features
    })

@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    """Return experiment metrics JSON."""
    metrics_file = METRICS_DIR / "all_metrics.json"
    if metrics_file.exists():
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return jsonify({"status": "ok", "metrics": data})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    return jsonify({"status": "empty", "metrics": {}})

@app.route("/api/distribution-shift", methods=["GET"])
def get_distribution_shift():
    """Return distribution shift statistical comparison between D1 and D2."""
    shift_file = METRICS_DIR / "distribution_shift.csv"
    if shift_file.exists():
        try:
            df = pd.read_csv(shift_file)
            records = df.to_dict(orient="records")
            return jsonify({"status": "ok", "data": records})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    return jsonify({"status": "empty", "data": []})

@app.route("/api/figures", methods=["GET"])
def get_figures():
    """Return categorized list of generated research figures."""
    if not FIGURES_DIR.exists():
        return jsonify({"figures": []})

    items = []
    for f in FIGURES_DIR.glob("*.png"):
        fname = f.name
        cat = "Evaluation"
        desc = "Research evaluation plot"
        
        if fname.startswith("cm_"):
            cat = "Confusion Matrix"
            desc = "Confusion Matrix showing classification distribution (TP, FP, TN, FN)"
        elif fname.startswith("roc_pr_"):
            cat = "ROC & PR Curves"
            desc = "Receiver Operating Characteristic and Precision-Recall Curves"
        elif "shift" in fname:
            cat = "Distribution Shift"
            desc = "Feature divergence analysis (Jensen-Shannon & Wasserstein shift)"
        elif "feat_imp" in fname or "shap" in fname:
            cat = "Explainability & SHAP"
            desc = "Feature importance ranking and SHAP tree-attribution"
        elif "gap" in fname or "blend" in fname or "metrics" in fname:
            cat = "Cross-Dataset Generalisation"
            desc = "Cross-dataset generalization gap & mitigation blending curve"

        items.append({
            "filename": fname,
            "url": f"/figures/{fname}",
            "category": cat,
            "description": desc,
            "size_kb": round(f.stat().st_size / 1024, 1)
        })

    # Sort by category then filename
    items.sort(key=lambda x: (x["category"], x["filename"]))
    return jsonify({"figures": items})

@app.route("/figures/<path:filename>")
def serve_figure(filename):
    """Serve figure image files."""
    return send_from_directory(FIGURES_DIR, filename)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    print(f"\n=======================================================")
    print(f"  🛡️ CyberShield Phishing URL Defense Platform")
    print(f"  Running at: http://{args.host}:{args.port}")
    print(f"=======================================================\n")
    app.run(host=args.host, port=args.port, debug=False)
