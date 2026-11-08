"""
main.py — CLI entry point for MalGuard Unified ML pipeline.

Usage:
    python main.py                  # Run full pipeline (train -> eval -> explain)
    python main.py --skip-train     # Evaluate & explain with existing models
    python main.py --classifier GradientBoosting  # Train a single classifier

Steps:
    1. Load / generate feature matrix
    2. Apply feature selection (Information Gain)
    3. Train all classifiers with 10-fold cross-validation + SMOTE
    4. Evaluate and generate comparison report
    5. Run SHAP explainability on best model
    6. Save reports and plots to reports/
"""

import argparse
import logging
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import config
from feature_extractor import FeatureExtractor
from feature_selector import FeatureSelector
from trainer import Trainer
from evaluator import Evaluator
from explainer import Explainer

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


# ── Synthetic data fallback ───────────────────────────────────────────────────
def _make_synthetic_dataset(n_samples: int = 2000) -> tuple[pd.DataFrame, pd.Series]:
    """
    Generate a reproducible synthetic feature matrix when no real APK dataset
    is available. Mirrors the 58-feature schema used by feature_extraction.py.
    """
    rng = np.random.default_rng(config.RANDOM_STATE)

    PERMISSIONS = [
        "android.permission.READ_CONTACTS", "android.permission.SEND_SMS",
        "android.permission.RECEIVE_SMS", "android.permission.READ_SMS",
        "android.permission.CAMERA", "android.permission.RECORD_AUDIO",
        "android.permission.ACCESS_FINE_LOCATION", "android.permission.ACCESS_COARSE_LOCATION",
        "android.permission.READ_CALL_LOG", "android.permission.WRITE_CALL_LOG",
        "android.permission.CALL_PHONE", "android.permission.PROCESS_OUTGOING_CALLS",
        "android.permission.READ_PHONE_STATE", "android.permission.GET_ACCOUNTS",
        "android.permission.USE_CREDENTIALS", "android.permission.MANAGE_ACCOUNTS",
        "android.permission.WRITE_CONTACTS", "android.permission.READ_EXTERNAL_STORAGE",
        "android.permission.WRITE_EXTERNAL_STORAGE", "android.permission.INTERNET",
        "android.permission.ACCESS_NETWORK_STATE", "android.permission.ACCESS_WIFI_STATE",
        "android.permission.CHANGE_WIFI_STATE", "android.permission.NFC",
        "android.permission.BLUETOOTH", "android.permission.VIBRATE",
        "android.permission.WAKE_LOCK", "android.permission.RECEIVE_BOOT_COMPLETED",
        "android.permission.SYSTEM_ALERT_WINDOW", "android.permission.INSTALL_PACKAGES",
    ]
    API_CALLS = [
        "getDeviceId", "getSubscriberId", "getLine1Number", "getNetworkOperator",
        "getSimSerialNumber", "sendTextMessage", "sendMultipartTextMessage",
        "execHttpRequest", "Runtime.exec", "ProcessBuilder.start",
        "loadUrl", "setJavaScriptEnabled", "addJavascriptInterface",
        "getInstalledPackages", "queryIntentActivities",
        "createSocket", "openConnection", "getContent",
        "openFileOutput", "deleteFile",
    ]
    INTENTS = [
        "android.intent.action.BOOT_COMPLETED",
        "android.intent.action.SMS_RECEIVED",
        "android.intent.action.PHONE_STATE",
        "android.intent.action.NEW_OUTGOING_CALL",
        "android.intent.action.PACKAGE_ADDED",
        "android.intent.action.PACKAGE_REMOVED",
        "android.intent.action.SEND",
        "android.intent.action.VIEW",
    ]

    all_features = PERMISSIONS + API_CALLS + INTENTS  # 58 features

    # Benign: sparser feature vectors
    n_benign  = n_samples // 2
    n_malware = n_samples - n_benign

    X_benign  = rng.binomial(1, 0.15, size=(n_benign,  len(all_features))).astype(float)
    X_malware = rng.binomial(1, 0.55, size=(n_malware, len(all_features))).astype(float)

    X = np.vstack([X_benign, X_malware])
    y = np.array([0] * n_benign + [1] * n_malware)

    # Shuffle
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]

    df = pd.DataFrame(X, columns=all_features)
    return df, pd.Series(y, name="label")


# ── Pipeline steps ────────────────────────────────────────────────────────────
def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    if os.path.exists(config.FEAT_FILE):
        logger.info("Loading existing feature matrix: %s", config.FEAT_FILE)
        df = pd.read_csv(config.FEAT_FILE)
        y  = df.pop("label")
        return df, y

    if os.path.exists(config.APK_DIR) and os.path.exists(config.LABEL_FILE):
        logger.info("Extracting features from APKs in: %s", config.APK_DIR)
        extractor = FeatureExtractor()
        df = extractor.extract_dataset(config.APK_DIR, config.LABEL_FILE)
        y  = df.pop("label")
        os.makedirs(config.DATA_DIR, exist_ok=True)
        combined = df.copy(); combined["label"] = y
        combined.to_csv(config.FEAT_FILE, index=False)
        logger.info("Feature matrix saved to: %s", config.FEAT_FILE)
        return df, y

    logger.warning("No dataset found — using synthetic data (2000 samples).")
    return _make_synthetic_dataset(2000)


def run_pipeline(args: argparse.Namespace) -> None:
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.REPORT_DIR, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("MalGuard Unified — ML Pipeline")
    logger.info("=" * 60)
    X, y = load_dataset()
    logger.info("Dataset: %d samples | %d features | %.1f%% malware",
                len(y), X.shape[1], y.mean() * 100)

    # ── 2. Feature selection ──────────────────────────────────────────────────
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE, stratify=y
    )

    logger.info("Applying Information-Gain feature selection (top-%d)…", config.TOP_K_FEATURES)
    selector = FeatureSelector(
        top_k=config.TOP_K_FEATURES,
        corr_threshold=config.CORR_THRESHOLD,
        variance_threshold=config.VARIANCE_THRESHOLD,
    )
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel  = selector.transform(X_test)
    logger.info("Selected %d features after IG + correlation filter.",
                len(selector.selected_features_))

    # ── 3. Train ──────────────────────────────────────────────────────────────
    if not args.skip_train:
        classifiers = ([args.classifier] if args.classifier
                       else config.CLASSIFIERS)
        logger.info("Training classifiers: %s", classifiers)
        trainer = Trainer(
            classifiers=classifiers,
            hyperparams=config.HYPERPARAMS,
            n_cv_folds=config.N_CV_FOLDS,
            use_smote=config.USE_SMOTE,
            smote_ratio=config.SMOTE_RATIO,
            random_state=config.RANDOM_STATE,
            model_dir=config.MODELS_DIR,
        )
        trainer.train(X_train_sel, y_train)
    else:
        logger.info("Skipping training (--skip-train).")

    # ── 4. Evaluate ───────────────────────────────────────────────────────────
    logger.info("Evaluating models…")
    evaluator = Evaluator(
        model_dir=config.MODELS_DIR,
        report_dir=config.REPORT_DIR,
    )
    results = evaluator.evaluate_all(X_test_sel, y_test)
    best_name = max(results, key=lambda k: results[k].get("roc_auc", 0))
    logger.info("Best model: %s (ROC-AUC=%.4f)",
                best_name, results[best_name]["roc_auc"])

    # ── 5. SHAP explainability ────────────────────────────────────────────────
    if config.RUN_SHAP:
        import joblib
        best_path = os.path.join(config.MODELS_DIR, f"{best_name}_best.pkl")
        if os.path.exists(best_path):
            logger.info("Running SHAP on %s…", best_name)
            model = joblib.load(best_path)
            explainer = Explainer(
                model=model,
                feature_names=selector.selected_features_,
                report_dir=config.REPORT_DIR,
                max_samples=config.SHAP_MAX_SAMPLES,
                top_features=config.SHAP_TOP_FEATURES,
            )
            explainer.explain(X_test_sel, y_test)
        else:
            logger.warning("Best model file not found; skipping SHAP.")

    logger.info("=" * 60)
    logger.info("Pipeline complete. Reports saved to: %s", config.REPORT_DIR)
    logger.info("=" * 60)


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="MalGuard Unified — Android Malware Detection ML Pipeline"
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip training; use existing saved models."
    )
    parser.add_argument(
        "--classifier", type=str, default=None,
        choices=config.CLASSIFIERS,
        help="Train a single classifier instead of all."
    )
    args = parser.parse_args()

    try:
        run_pipeline(args)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(0)
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
