"""
config.py — Global configuration for MalGuard Unified.

Merges Project A (ML pipeline) and Project B (Crypto/Hash) settings.
IEEE Base Paper: "A Robust and Efficient Machine Learning Framework for
Enhancing Early Detection of Android Malware" (IEEE Access, 2025)
"""

import os

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "model")       # Streamlit model dir
MODELS_DIR = os.path.join(BASE_DIR, "models")      # CLI training output dir
REPORT_DIR = os.path.join(BASE_DIR, "reports")     # evaluation reports & plots

# Project-A style paths (used by CLI pipeline)
APK_DIR    = os.path.join(DATA_DIR, "apks")
LABEL_FILE = os.path.join(DATA_DIR, "labels.csv")
FEAT_FILE  = os.path.join(DATA_DIR, "features.csv")

# Project-B style paths (used by Streamlit app)
HASH_DB_FILE       = os.path.join(DATA_DIR, "hash_database.csv")
BASELINES_FILE     = os.path.join(DATA_DIR, "trusted_baselines.json")
SCAN_HISTORY_DB    = os.path.join(DATA_DIR, "scan_history.db")
MODEL_FILE         = os.path.join(MODEL_DIR, "GradientBoosting_best.pkl")

# ── Dataset ──────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.2    # 80/20 split
N_CV_FOLDS   = 10    # stratified k-fold

# ── Feature Selection ────────────────────────────────────────────────────────
TOP_K_FEATURES      = 100
CORR_THRESHOLD      = 0.95
VARIANCE_THRESHOLD  = 0.005

# ── Class Imbalance ──────────────────────────────────────────────────────────
USE_SMOTE   = True
SMOTE_RATIO = 1.0

# ── Classifiers ──────────────────────────────────────────────────────────────
CLASSIFIERS = [
    "RandomForest",
    "GradientBoosting",
    "AdaBoost",
    "SVM",
    "KNN",
    "NaiveBayes",
]

# ── Hyperparameters ──────────────────────────────────────────────────────────
HYPERPARAMS = {
    "RandomForest": {
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_split": 2,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    },
    "GradientBoosting": {
        "n_estimators": 150,
        "learning_rate": 0.1,
        "max_depth": 5,
        "random_state": RANDOM_STATE,
    },
    "AdaBoost": {
        "n_estimators": 100,
        "learning_rate": 1.0,
        "random_state": RANDOM_STATE,
    },
    "SVM": {
        "kernel": "rbf",
        "C": 10,
        "gamma": "scale",
        "probability": True,
    },
    "KNN": {
        "n_neighbors": 5,
        "metric": "minkowski",
        "n_jobs": -1,
    },
    "NaiveBayes": {
        "var_smoothing": 1e-9,
    },
}

# ── XAI (SHAP) ───────────────────────────────────────────────────────────────
RUN_SHAP          = True
SHAP_MAX_SAMPLES  = 100
SHAP_TOP_FEATURES = 15

# ── Risk Scoring Thresholds ──────────────────────────────────────────────────
# Score components (additive):
#   +80  known malware hash in DB
#   +40  ML model predicts malware
#   +20  integrity mismatch detected
#   +10  invalid / unsigned APK signature
RISK_HIGH   = 70
RISK_MEDIUM = 40

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
