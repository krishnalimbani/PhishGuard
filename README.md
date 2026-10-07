# 🛡️ PhishGuard – Phishing URL Detection Web App

> **BCA Mini Project** | Python · Flask · Scikit-learn · TensorFlow · SQLite

PhishGuard analyses the *structure* of a URL and predicts whether it is **Phishing** or **Legitimate** using two independently trained models — a **Random Forest** (ML) and a **Neural Network** (Deep Learning). No URL is ever visited; analysis is completely static.

---

## 📁 Project Structure

```
PhishGuard/
├── app.py                  # Flask web application
├── train_model.py          # Model training script (run once)
├── requirements.txt        # Python dependencies
│
├── data/
│   └── phishing_urls.csv   # Dataset (auto-downloaded or synthetic)
│
├── models/
│   ├── rf_model.pkl        # Saved Random Forest model
│   ├── scaler.pkl          # Feature scaler
│   ├── nn_model.keras      # Saved Neural Network model
│   └── metrics.json        # Evaluation metrics
│
├── utils/
│   ├── __init__.py
│   ├── feature_extractor.py  # URL feature engineering
│   └── database.py           # SQLite helpers
│
├── templates/
│   └── index.html          # Single-page dashboard
│
├── static/
│   ├── css/style.css       # Dark cybersecurity theme
│   ├── js/main.js          # Frontend logic + charts
│   └── images/             # Confusion matrix PNGs (generated)
│
└── database/
    └── phishguard.db       # SQLite scan history (auto-created)
```

---

## ⚙️ Installation

### 1. Prerequisites
- Python 3.9 – 3.11 (TensorFlow 2.13 requires < 3.12)
- `pip`

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** TensorFlow installation can take a few minutes.

---

## 📦 Dataset Setup

The training script handles this automatically:

| Option | What happens |
|--------|-------------|
| **Internet available** | Downloads the *PhiUSIIL Phishing URL Dataset* (~235 k rows) from the UCI ML Repository |
| **Offline / download fails** | Generates a 4 000-row synthetic dataset so the project works without internet |

You can also place your own CSV at `data/phishing_urls.csv` with columns **`url`** and **`label`** (0 = legitimate, 1 = phishing).

---

## 🏋️ Model Training

Run **once** before starting the web app:

```bash
python train_model.py
```

What it does:
1. Downloads or generates the dataset
2. Extracts 25 URL-based features for every sample
3. Trains a **Random Forest** (150 trees)
4. Trains a **Neural Network** (4 dense layers, early stopping)
5. Saves both models to `models/`
6. Saves evaluation metrics to `models/metrics.json`
7. Saves confusion matrix images to `static/images/`

Expected output (approximate, depends on dataset):
```
Random Forest  →  Accuracy: 0.9650  F1: 0.9650
Neural Network →  Accuracy: 0.9580  F1: 0.9575
```

---

## 🚀 Running the App

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

---

## 🔍 Features Extracted from URLs

| Feature | Description |
|---------|-------------|
| `url_length` | Total character count |
| `num_dots` | Number of `.` in the URL |
| `num_hyphens` | Number of `-` |
| `num_underscores` | Number of `_` |
| `num_slashes` | Number of `/` |
| `num_question_marks` | Number of `?` |
| `num_equal_signs` | Number of `=` |
| `num_at_signs` | Number of `@` |
| `num_ampersands` | Number of `&` |
| `num_percent` | Number of `%` (URL-encoded chars) |
| `has_https` | 1 if scheme is `https`, else 0 |
| `has_ip_address` | 1 if host is an IPv4 address |
| `num_subdomains` | Subdomain depth (parts − 2) |
| `domain_length` | Length of the hostname |
| `path_length` | Length of the URL path |
| `num_params` | Number of query-string parameters |
| `num_fragments` | 1 if fragment (`#`) present |
| `num_digits_in_domain` | Digit count in hostname |
| `num_suspicious_keywords` | Matches against a phishing keyword list |
| `has_port` | 1 if a non-standard port is specified |
| `num_special_chars` | Count of unusual special characters |
| `entropy` | Shannon entropy of the full URL |
| `tld_length` | Length of the top-level domain |
| `num_digits` | Total digit count in URL |
| `double_slash_redirect` | 1 if `//` appears in the path |

---

## 🧠 Model Architecture

### Random Forest
- **Library:** Scikit-learn
- **Trees:** 150, max depth 20
- **Input:** 25 numeric features (no scaling needed)

### Neural Network
```
Input(25)  →  Dense(128, ReLU) → BatchNorm → Dropout(0.3)
           →  Dense(64,  ReLU) → BatchNorm → Dropout(0.2)
           →  Dense(32,  ReLU)
           →  Dense(1, Sigmoid)
```
- **Loss:** Binary cross-entropy
- **Optimizer:** Adam (lr = 0.001)
- **Input:** Scaled features (StandardScaler)
- **Early stopping:** patience = 5 epochs

### Ensemble
Final prediction = majority vote of RF and NN; confidence breaks ties.

---

## 📊 Dashboard Sections

| Tab | Description |
|-----|-------------|
| **Dashboard** | URL scanner + real-time result + stats |
| **Model Performance** | Accuracy / Precision / Recall / F1 charts + confusion matrices |
| **Scan History** | Last 50 scanned URLs stored in SQLite |
| **About** | Explanation of phishing and how PhishGuard works |

---

## ⚠️ Disclaimer

PhishGuard performs **static structural analysis only**. It never visits, loads, or connects to any submitted URL. It is an educational project and should not be used as the sole means of determining whether a website is safe.

---

## 👨‍🎓 Credits

Built as a **BCA Mini Project** to demonstrate the application of Machine Learning and Deep Learning in the domain of Cybersecurity.
