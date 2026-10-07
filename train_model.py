"""
train_model.py
--------------
1. Downloads / loads the phishing URL dataset.
2. Extracts URL-based features.
3. Trains a Random Forest (ML) model.
4. Trains a Neural Network (Keras) model.
5. Saves both models + the scaler + evaluation metrics to disk.

Dataset used:
  PhiUSIIL Phishing URL Dataset (UCI ML Repository)
  URL: https://archive.ics.uci.edu/ml/machine-learning-databases/00967/PhiUSIIL_Phishing_URL_Dataset.csv
  If the download fails, the script falls back to a small synthetic dataset
  so the project still works offline.

Run once:
    python train_model.py
"""

import os
import sys
import json
import warnings
import urllib.request

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")          # headless backend – no display required
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # suppress TF C++ logs
import tensorflow as tf
from tensorflow import keras

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(DATA_DIR,  exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "images"), exist_ok=True)

DATASET_PATH = os.path.join(DATA_DIR, "phishing_urls.csv")

DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases"
    "/00967/PhiUSIIL_Phishing_URL_Dataset.csv"
)

# ─────────────────────────────────────────────
# Feature extraction import
# ─────────────────────────────────────────────
sys.path.insert(0, BASE_DIR)
from utils.feature_extractor import features_to_list, FEATURE_NAMES


# ─────────────────────────────────────────────
# 1.  Load dataset
# ─────────────────────────────────────────────
def download_dataset():
    print("[INFO] Attempting to download dataset …")
    try:
        urllib.request.urlretrieve(DATASET_URL, DATASET_PATH)
        print("[INFO] Dataset downloaded successfully.")
        return True
    except Exception as exc:
        print(f"[WARN] Download failed: {exc}")
        return False


def build_synthetic_dataset(n=4000):
    """
    Create a small labelled dataset by building realistic-looking
    phishing and legitimate URLs, then extracting features.
    """
    print("[INFO] Building synthetic dataset …")
    import random, string

    legitimate_templates = [
        "https://www.{}.com/",
        "https://{}.org/about",
        "https://shop.{}.co.uk/products",
        "https://blog.{}.net/post/{}",
        "https://www.{}.edu/courses",
    ]
    phishing_templates = [
        "http://{}-login.com/verify?user={}&token={}",
        "http://{}.free-update.xyz/account/confirm",
        "http://192.168.{}.{}/paypal/signin",
        "http://secure-{}-login.tk/update?id={}",
        "http://www.{}.com.{}.ru/signin",
        "http://{}-alert.info/winner/claim?prize={}",
    ]

    def rand_word(n=6):
        return "".join(random.choices(string.ascii_lowercase, k=n))

    rows = []
    half = n // 2
    for _ in range(half):
        tpl = random.choice(legitimate_templates)
        url = tpl.format(*[rand_word() for _ in range(tpl.count("{}"))])
        rows.append((url, 0))   # 0 = legitimate

    for _ in range(half):
        tpl = random.choice(phishing_templates)
        url = tpl.format(*[rand_word() for _ in range(tpl.count("{}"))])
        rows.append((url, 1))   # 1 = phishing

    random.shuffle(rows)
    df = pd.DataFrame(rows, columns=["url", "label"])
    df.to_csv(DATASET_PATH, index=False)
    print(f"[INFO] Synthetic dataset saved ({len(df)} rows).")
    return df


def load_dataset():
    if os.path.exists(DATASET_PATH):
        print(f"[INFO] Loading existing dataset from {DATASET_PATH}")
        df = pd.read_csv(DATASET_PATH, low_memory=False)
    else:
        if not download_dataset():
            return build_synthetic_dataset()
        df = pd.read_csv(DATASET_PATH, low_memory=False)

    # Normalise column names to lowercase
    df.columns = [c.lower().strip() for c in df.columns]

    # Try to identify the URL and label columns
    url_col   = next((c for c in df.columns if "url" in c), None)
    label_col = next((c for c in df.columns
                      if c in ("label", "class", "result", "phishing", "type")), None)

    if url_col is None or label_col is None:
        print("[WARN] Could not identify columns; building synthetic dataset.")
        return build_synthetic_dataset()

    df = df[[url_col, label_col]].dropna()
    df.columns = ["url", "label"]

    # Map label to 0/1
    unique_labels = df["label"].unique()
    if set(unique_labels) == {0, 1} or set(unique_labels) == {"0", "1"}:
        df["label"] = df["label"].astype(int)
    elif set(unique_labels) <= {-1, 0, 1}:
        # Some datasets use -1 for phishing, 1 for legitimate
        df["label"] = df["label"].apply(lambda x: 1 if int(x) == -1 else 0)
    else:
        mapping = {v: i for i, v in enumerate(sorted(unique_labels))}
        df["label"] = df["label"].map(mapping)

    # Cap at 30 000 rows for reasonable training time
    if len(df) > 30000:
        df = df.sample(30000, random_state=42)

    print(f"[INFO] Dataset loaded: {len(df)} rows  |  "
          f"Legitimate={sum(df['label']==0)}  Phishing={sum(df['label']==1)}")
    return df


# ─────────────────────────────────────────────
# 2.  Build feature matrix
# ─────────────────────────────────────────────
def build_features(df):
    print("[INFO] Extracting features (this may take a moment) …")
    X = []
    y = []
    for _, row in df.iterrows():
        try:
            feats = features_to_list(str(row["url"]))
            X.append(feats)
            y.append(int(row["label"]))
        except Exception:
            pass
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"[INFO] Feature matrix shape: {X.shape}")
    return X, y


# ─────────────────────────────────────────────
# 3.  Train Random Forest
# ─────────────────────────────────────────────
def train_random_forest(X_train, y_train, X_test, y_test):
    print("[INFO] Training Random Forest …")
    clf = RandomForestClassifier(
        n_estimators=150, max_depth=20,
        random_state=42, n_jobs=-1
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred),  4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred,    zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred,         zero_division=0), 4),
    }
    cm = confusion_matrix(y_test, y_pred)
    print("[ML] Random Forest metrics:", metrics)
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"]))
    return clf, metrics, cm


# ─────────────────────────────────────────────
# 4.  Train Neural Network
# ─────────────────────────────────────────────
def train_neural_network(X_train, y_train, X_test, y_test):
    print("[INFO] Training Neural Network …")
    n_features = X_train.shape[1]

    model = keras.Sequential([
        keras.layers.Input(shape=(n_features,)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=5, restore_best_weights=True, monitor="val_loss"
        ),
        keras.callbacks.ReduceLROnPlateau(
            factor=0.5, patience=3, monitor="val_loss", min_lr=1e-6
        ),
    ]

    model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=64,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
    )

    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred),  4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred,    zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred,         zero_division=0), 4),
    }
    cm = confusion_matrix(y_test, y_pred)
    print("[NN] Neural Network metrics:", metrics)
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"]))
    return model, metrics, cm


# ─────────────────────────────────────────────
# 5.  Save charts
# ─────────────────────────────────────────────
def save_confusion_matrix(cm, title, filename):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Legitimate", "Phishing"],
        yticklabels=["Legitimate", "Phishing"],
        ax=ax,
    )
    ax.set_title(title)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    out = os.path.join(STATIC_DIR, "images", filename)
    plt.savefig(out, dpi=100)
    plt.close()
    print(f"[INFO] Saved {out}")


# ─────────────────────────────────────────────
# 6.  Main
# ─────────────────────────────────────────────
def main():
    # Load
    df = load_dataset()

    # Features
    X, y = build_features(df)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Train ML
    rf_model, rf_metrics, rf_cm = train_random_forest(
        X_train, y_train, X_test, y_test
    )

    # Train NN
    nn_model, nn_metrics, nn_cm = train_neural_network(
        X_train_s, y_train, X_test_s, y_test
    )

    # Save models
    joblib.dump(rf_model, os.path.join(MODEL_DIR, "rf_model.pkl"))
    joblib.dump(scaler,   os.path.join(MODEL_DIR, "scaler.pkl"))
    nn_model.save(os.path.join(MODEL_DIR, "nn_model.keras"))

    # Save metrics
    all_metrics = {"random_forest": rf_metrics, "neural_network": nn_metrics}
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    print("[INFO] Metrics saved.")

    # Save confusion matrices
    save_confusion_matrix(rf_cm,  "Random Forest – Confusion Matrix",  "rf_cm.png")
    save_confusion_matrix(nn_cm,  "Neural Network – Confusion Matrix", "nn_cm.png")

    print("\n" + "="*55)
    print("  Training complete!")
    print(f"  Random Forest   Accuracy: {rf_metrics['accuracy']:.4f}  "
          f"F1: {rf_metrics['f1']:.4f}")
    print(f"  Neural Network  Accuracy: {nn_metrics['accuracy']:.4f}  "
          f"F1: {nn_metrics['f1']:.4f}")
    print("="*55)


if __name__ == "__main__":
    main()
