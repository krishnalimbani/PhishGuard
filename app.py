"""
app.py  –  PhishGuard Flask application
"""

import os
import sys
import json

import numpy as np
from flask import Flask, render_template, request, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from utils.feature_extractor import extract_features, features_to_list, FEATURE_NAMES
from utils.database import init_db, save_scan, get_all_scans, get_stats

app = Flask(__name__)

# ──────────────────────────────────────────────────────
# Load trained models (loaded once at startup)
# ──────────────────────────────────────────────────────
MODEL_DIR = os.path.join(BASE_DIR, "models")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

rf_model  = None
scaler    = None
nn_model  = None
metrics   = {}

def load_models():
    """Load RF + scaler immediately; defer NN until first scan request."""
    global rf_model, scaler, metrics

    rf_path     = os.path.join(MODEL_DIR, "rf_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    nn_path     = os.path.join(MODEL_DIR, "nn_model.keras")

    missing = [p for p in (rf_path, scaler_path, nn_path) if not os.path.exists(p)]
    if missing:
        app.logger.warning("Models not found: %s – run train_model.py first.", missing)
        return False

    try:
        import joblib
        rf_model = joblib.load(rf_path)
        scaler   = joblib.load(scaler_path)

        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH) as f:
                metrics = json.load(f)

        app.logger.info("RF + scaler loaded. NN will load on first scan.")
        return True
    except Exception as exc:
        app.logger.error("Error loading models: %s", exc)
        return False


def _ensure_nn():
    """Load the Neural Network on first use (lazy load)."""
    global nn_model
    if nn_model is not None:
        return True
    nn_path = os.path.join(MODEL_DIR, "nn_model.keras")
    try:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
        import tensorflow as tf
        nn_model = tf.keras.models.load_model(nn_path)
        return True
    except Exception as exc:
        app.logger.error("Error loading NN model: %s", exc)
        return False


models_ready = False


# ──────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────

@app.route("/")
def index():
    stats = get_stats()
    ml_acc = metrics.get("random_forest", {}).get("accuracy", 0)
    return render_template("index.html",
                           stats=stats,
                           models_ready=models_ready,
                           ml_accuracy=ml_acc)


@app.route("/api/scan", methods=["POST"])
def scan_url():
    """Scan a URL and return prediction results as JSON."""
    data = request.get_json(force=True)
    url  = (data.get("url") or "").strip()

    if not url:
        return jsonify({"error": "No URL provided."}), 400

    # Guard: refuse obviously dangerous inputs (no network calls ever made)
    if len(url) > 2000:
        return jsonify({"error": "URL too long (max 2000 chars)."}), 400

    if not models_ready:
        return jsonify({"error": "Models not trained yet. Run train_model.py first."}), 503

    if not _ensure_nn():
        return jsonify({"error": "Neural Network model failed to load."}), 503

    # Extract features
    feature_dict = extract_features(url)
    feature_list = [feature_dict[n] for n in FEATURE_NAMES]
    X = np.array([feature_list], dtype=np.float32)

    # ── ML prediction ──
    ml_proba   = rf_model.predict_proba(X)[0]          # [prob_legit, prob_phish]
    ml_class   = int(np.argmax(ml_proba))
    ml_conf    = float(ml_proba[ml_class])
    ml_label   = "Phishing" if ml_class == 1 else "Legitimate"

    # ── NN prediction ──
    X_scaled   = scaler.transform(X)
    nn_proba   = float(nn_model.predict(X_scaled, verbose=0)[0][0])
    nn_class   = int(nn_proba >= 0.5)
    nn_label   = "Phishing" if nn_class == 1 else "Legitimate"
    nn_conf    = nn_proba if nn_class == 1 else 1 - nn_proba

    # ── Ensemble decision (majority vote with confidence tiebreak) ──
    if ml_class == nn_class:
        final_result = ml_label
    else:
        final_result = ml_label if ml_conf >= nn_conf else nn_label

    # ── Save to DB ──
    save_scan(url, final_result, ml_label, ml_conf, nn_label, nn_conf)

    # ── Format features for display ──
    feature_display = [
        {"name": name.replace("_", " ").title(), "value": value}
        for name, value in feature_dict.items()
    ]

    # ── Simple explanation ──
    reasons = []
    if feature_dict["has_https"] == 0:
        reasons.append("URL does not use HTTPS (insecure).")
    if feature_dict["has_ip_address"] == 1:
        reasons.append("URL uses an IP address instead of a domain name.")
    if feature_dict["num_suspicious_keywords"] > 0:
        reasons.append(f"Contains {feature_dict['num_suspicious_keywords']} suspicious keyword(s).")
    if feature_dict["num_subdomains"] > 2:
        reasons.append(f"Unusually high number of subdomains ({feature_dict['num_subdomains']}).")
    if feature_dict["url_length"] > 75:
        reasons.append(f"URL is very long ({feature_dict['url_length']} chars).")
    if feature_dict["num_hyphens"] > 3:
        reasons.append(f"Many hyphens ({feature_dict['num_hyphens']}) in URL.")
    if feature_dict["double_slash_redirect"] == 1:
        reasons.append("URL contains double-slash redirect pattern.")
    if not reasons:
        reasons.append("No obvious phishing indicators found in URL structure.")

    return jsonify({
        "url":     url,
        "result":  final_result,
        "ml": {
            "prediction": ml_label,
            "confidence": round(ml_conf * 100, 2),
        },
        "nn": {
            "prediction": nn_label,
            "confidence": round(nn_conf * 100, 2),
        },
        "features": feature_display,
        "reasons":  reasons,
    })


@app.route("/api/history")
def history():
    """Return recent scan history as JSON."""
    scans = get_all_scans(limit=50)
    return jsonify(scans)


@app.route("/api/stats")
def stats():
    """Return aggregate scan stats."""
    s = get_stats()
    return jsonify(s)


@app.route("/api/metrics")
def model_metrics():
    """Return model performance metrics."""
    return jsonify(metrics)


# ──────────────────────────────────────────────────────
# Startup
# ──────────────────────────────────────────────────────

if __name__ == "__main__":
    init_db()
    models_ready = load_models()
    app.run(debug=False, host="0.0.0.0", port=5000)
else:
    # When run via `flask run`
    init_db()
    models_ready = load_models()
