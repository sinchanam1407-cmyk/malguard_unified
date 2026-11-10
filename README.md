# MalGuard Unified 🛡️

> AI-Based Android Malware Detection with Cryptographic Hash Verification

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=sinchanam1407-cmyk/malguard_unified&branch=main&mainModule=app.py)

## 🚀 One-Click Deploy

Click the button above **OR** go to:

👉 **[share.streamlit.io](https://share.streamlit.io)** → New App → `sinchanam1407-cmyk/malguard_unified` → `app.py` → Deploy

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Model

```
https://github.com/sinchanam1407-cmyk/malguard_unified/raw/main/model/GradientBoosting_best.pkl
```

## Performance

| Model | Accuracy | ROC-AUC |
|---|---|---|
| **Gradient Boosting** | **99.23%** | **99.87%** |
| Random Forest | 98.47% | 99.62% |
| SVM | 96.92% | 98.76% |

## Tests

```bash
pytest tests/ -v   # 39/39 PASS
```
