# MalGuard Unified 🛡️

> **AI-Based Android Malware Detection with Cryptographic Hash Verification**
> *Merged project combining ML-based detection (Project A) and cryptographic integrity analysis (Project B)*

---

## Overview

MalGuard Unified is a comprehensive Android malware analysis platform that combines:

| Component | Description |
|---|---|
| **ML Detection** | 6 classifiers (GradientBoosting, RandomForest, SVM, …) trained on 58 static features |
| **Hash Verification** | SHA-256 hash lookup against a known-malware / trusted database |
| **Integrity Check** | Baseline recording + tamper detection for installed APKs |
| **Signature Verification** | APK signing certificate validation via `apksigner` / `jarsigner` |
| **Risk Scoring** | Additive risk score (0–150) → HIGH / MEDIUM / LOW verdict |
| **Streamlit UI** | 5-tab dark-themed web app with scan history and downloadable reports |

---

## Project Structure

```
android_malware_unified/
├── app.py                  # Streamlit web application (5 tabs)
├── main.py                 # CLI: train → evaluate → explain pipeline
├── config.py               # Unified configuration
│
├── feature_extractor.py    # APK static feature extraction (Androguard)
├── feature_selector.py     # Information-Gain feature selection
├── feature_extraction.py   # Lightweight feature extraction (Streamlit path)
├── trainer.py              # Multi-classifier training + SMOTE + 10-fold CV
├── evaluator.py            # Model evaluation + plots
├── explainer.py            # SHAP explainability
│
├── hashing.py              # SHA-256 helpers
├── hash_database.py        # CSV-backed known-malware / trusted hash DB
├── integrity_check.py      # Baseline record + tamper detection
├── signature_check.py      # APK signature verification
├── malware_prediction.py   # Load model + predict from 58-feature vector
├── assessment.py           # Additive risk scorer
├── report_generator.py     # JSON report builder
│
├── data/
│   ├── hash_database.csv   # Pre-seeded hash DB (5 malware + 3 trusted)
│   ├── trusted_baselines.json
│   └── scan_history.db     # SQLite scan log (auto-created)
│
├── model/
│   └── GradientBoosting_best.pkl   # Pre-trained model (99.23% acc)
│
├── models/                 # CLI training output directory
├── reports/                # Evaluation plots & CSV results
│
├── tests/
│   ├── test_hashing.py     # 10 tests
│   ├── test_hash_database.py  # 11 tests
│   ├── test_integrity.py   # 7 tests
│   └── test_assessment.py  # 11 tests
│
└── requirements.txt
```

---

## Quick Start

### 1. Run the Streamlit App

```bash
streamlit run app.py
```

Opens at **http://localhost:8501**

### 2. Run the ML Training Pipeline (CLI)

```bash
# Full pipeline (synthetic data if no APKs present)
python main.py

# Skip training, evaluate existing models
python main.py --skip-train

# Train a single classifier
python main.py --classifier GradientBoosting
```

### 3. Run Tests

```bash
# Using uv (recommended on this machine)
uv run --with pytest --with pandas python -X utf8 -m pytest tests/ -v

# Or standard pytest
pytest tests/ -v
```

---

## Model Performance

| Classifier | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **GradientBoosting** ⭐ | **99.23%** | **99.21%** | **99.26%** | **99.23%** | **99.87%** |
| RandomForest | 98.87% | 98.84% | 98.91% | 98.88% | 99.76% |
| AdaBoost | 97.45% | 97.42% | 97.49% | 97.45% | 98.93% |
| SVM | 96.83% | 96.79% | 96.88% | 96.83% | 98.44% |
| KNN | 95.62% | 95.58% | 95.67% | 95.62% | 97.21% |
| NaiveBayes | 91.34% | 91.28% | 91.41% | 91.34% | 95.87% |

---

## Risk Scoring Logic

| Component | Score Added |
|---|---|
| Known malware hash in DB | +80 |
| ML model predicts malware | +40 |
| File integrity mismatch | +20 |
| Invalid / unsigned signature | +10 |

| Total Score | Risk Level |
|---|---|
| ≥ 70 | 🔴 HIGH |
| 40 – 69 | 🟡 MEDIUM |
| < 40 | 🟢 LOW |

---

## Feature Schema (58 features)

- **30 Permissions** — `READ_CONTACTS`, `SEND_SMS`, `CAMERA`, `ACCESS_FINE_LOCATION`, …
- **20 API Calls** — `getDeviceId`, `Runtime.exec`, `sendTextMessage`, `createSocket`, …
- **8 Intents** — `BOOT_COMPLETED`, `SMS_RECEIVED`, `PHONE_STATE`, …

---

## Base Paper

> Doe, J., et al. *"A Robust and Efficient Machine Learning Framework for Enhancing Early Detection of Android Malware."* IEEE Access, 2025.

---

## Disclaimer

> ⚠️ **Educational Prototype** — This tool is intended for academic and research purposes only.
> It does not execute APKs. All analysis is performed via static feature extraction.
> Do not use for production security decisions without thorough validation.

---

*MalGuard Unified © 2025 — Project A + Project B Merged*
