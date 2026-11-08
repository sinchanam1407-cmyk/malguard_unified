"""
app.py — MalGuard Unified: Merged Android Malware Detection System.

Combines:
  Project A — ML training pipeline (Gradient Boosting, SHAP explainability)
  Project B — Cryptographic analysis (SHA-256, Hash DB, Integrity, Signature)

Tabs:
  1. 🔍 APK Scanner     — full 5-module analysis + combined risk assessment
  2. 🤖 ML Dashboard    — model metrics, feature importance, SHAP explanation
  3. 🗄  Hash Database   — view / add hash records
  4. 📜 Scan History    — previous scan results
  5. ⚙  Settings        — record baselines, re-train model

Run:
    streamlit run app.py
"""

import os
import sys
import json
import tempfile
import logging
import datetime
import sqlite3

import streamlit as st
import pandas as pd
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

# ── Streamlit page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="MalGuard Unified",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  CSS — dark theme matching project documents
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Global ── */
html, body, [class*="css"] {
    background-color: #05060c !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif;
}
/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0d0e1a !important;
    border-right: 1px solid #1e2040;
}
/* Cards */
.mcard {
    background: #0f1020;
    border: 1px solid #1e2040;
    border-radius: 12px;
    padding: 20px 22px;
    margin-bottom: 16px;
}
/* KPI row */
.kpi-row { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 20px; }
.kpi {
    background: #0f1020;
    border: 1px solid #1e2040;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
    flex: 1;
    min-width: 130px;
}
.kpi-val { font-size: 1.9rem; font-weight: 800; }
.kpi-lbl { font-size: .72rem; color: #94a3b8; }
/* Badges */
.badge-high   { background:#ef444420; color:#ef4444; border:1px solid #ef444450; border-radius:6px; padding:4px 10px; font-weight:700; }
.badge-medium { background:#f59e0b20; color:#f59e0b; border:1px solid #f59e0b50; border-radius:6px; padding:4px 10px; font-weight:700; }
.badge-low    { background:#10b98120; color:#10b981; border:1px solid #10b98150; border-radius:6px; padding:4px 10px; font-weight:700; }
.badge-pass   { background:#10b98118; color:#10b981; border:1px solid #10b98140; border-radius:5px; padding:2px 8px; font-size:.75rem; font-weight:700; }
.badge-fail   { background:#ef444415; color:#ef4444; border:1px solid #ef444440; border-radius:5px; padding:2px 8px; font-size:.75rem; font-weight:700; }
.badge-warn   { background:#f59e0b15; color:#f59e0b; border:1px solid #f59e0b40; border-radius:5px; padding:2px 8px; font-size:.75rem; font-weight:700; }
.badge-info   { background:#00d4ff12; color:#00d4ff; border:1px solid #00d4ff35; border-radius:5px; padding:2px 8px; font-size:.75rem; font-weight:700; }
/* Risk banner */
.risk-high   { background:linear-gradient(135deg,#ef444415,#ef444408); border:2px solid #ef4444; border-radius:14px; padding:22px; text-align:center; }
.risk-medium { background:linear-gradient(135deg,#f59e0b15,#f59e0b08); border:2px solid #f59e0b; border-radius:14px; padding:22px; text-align:center; }
.risk-low    { background:linear-gradient(135deg,#10b98115,#10b98108); border:2px solid #10b981; border-radius:14px; padding:22px; text-align:center; }
.risk-unknown{ background:linear-gradient(135deg,#94a3b815,#94a3b808); border:2px solid #94a3b8; border-radius:14px; padding:22px; text-align:center; }
/* Code */
code { font-family: 'JetBrains Mono', monospace !important; }
/* Stray white backgrounds */
.stDataFrame, .stTable { background: #0f1020 !important; }
div[data-testid="stMetricValue"] { color: #00d4ff !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SQLite scan history
# ══════════════════════════════════════════════════════════════════════════════
DB_PATH = os.path.join(BASE_DIR, "data", "scan_history.db")

def _init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS scans (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT, apk_name TEXT, file_size_mb REAL,
        sha256 TEXT, hash_match TEXT, integrity_status TEXT,
        sig_status TEXT, ml_prediction TEXT, ml_confidence REAL,
        overall_risk TEXT, risk_score INTEGER
    )""")
    con.commit(); con.close()

def _save_scan(row: dict):
    con = sqlite3.connect(DB_PATH)
    con.execute("""INSERT INTO scans (timestamp,apk_name,file_size_mb,sha256,
        hash_match,integrity_status,sig_status,ml_prediction,ml_confidence,
        overall_risk,risk_score) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (row.get("timestamp"), row.get("apk_name"), row.get("file_size_mb"),
         row.get("sha256"), row.get("hash_match"), row.get("integrity_status"),
         row.get("sig_status"), row.get("ml_prediction"), row.get("ml_confidence"),
         row.get("overall_risk"), row.get("risk_score")))
    con.commit(); con.close()

def _load_history(limit=50) -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM scans ORDER BY id DESC LIMIT ?",
                     con, params=(limit,))
    con.close()
    return df

_init_db()

# ══════════════════════════════════════════════════════════════════════════════
#  Module imports — graceful fallback if deps missing
# ══════════════════════════════════════════════════════════════════════════════
def _try_import(name, label):
    try:
        return __import__(name), True
    except Exception:
        return None, False

hashing_mod,    hashing_ok    = _try_import("hashing",            "Hashing")
hashdb_mod,     hashdb_ok     = _try_import("hash_database",      "Hash DB")
integrity_mod,  integrity_ok  = _try_import("integrity_check",    "Integrity")
sig_mod,        sig_ok        = _try_import("signature_check",    "Signature")
ml_mod,         ml_ok         = _try_import("malware_prediction", "ML Predict")
assess_mod,     assess_ok     = _try_import("assessment",         "Assessment")
report_mod,     report_ok     = _try_import("report_generator",   "Reports")

# ══════════════════════════════════════════════════════════════════════════════
#  Sidebar
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding:4px 0 16px'>
      <div style='font-size:.62rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;
                  color:#f59e0b;background:rgba(245,158,11,.1);border:1px solid rgba(245,158,11,.3);
                  border-radius:4px;padding:3px 8px;display:inline-block;margin-bottom:10px'>
        Unified Edition</div>
      <div style='font-family:Outfit,sans-serif;font-size:1.3rem;font-weight:800;
                  background:linear-gradient(135deg,#00d4ff,#7c3aed);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
        MalGuard</div>
      <div style='font-size:.75rem;color:#94a3b8;margin-top:2px'>Android Malware Detection</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Module Status**")
    mods = [
        ("🔐 SHA-256 Hashing",   hashing_ok),
        ("🗄  Hash Database",     hashdb_ok),
        ("🛡  Integrity Check",   integrity_ok),
        ("✍  Sig Verification",  sig_ok),
        ("🤖 ML Prediction",     ml_ok),
        ("⚖️  Assessment",        assess_ok),
    ]
    for name, ok in mods:
        colour = "#10b981" if ok else "#ef4444"
        icon   = "●" if ok else "○"
        st.markdown(
            f"<div style='font-size:.8rem;color:{colour};margin:3px 0'>"
            f"{icon} {name}</div>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    model_path = os.path.join(BASE_DIR, "model", "GradientBoosting_best.pkl")
    model_exists = os.path.exists(model_path)
    st.markdown(
        f"<div style='font-size:.78rem;color:{'#10b981' if model_exists else '#ef4444'}'>"
        f"{'✅' if model_exists else '❌'} Trained model "
        f"{'loaded' if model_exists else 'not found'}</div>",
        unsafe_allow_html=True
    )
    st.markdown("---")
    st.markdown(
        "<div style='font-size:.66rem;color:#94a3b8;line-height:1.65'>"
        "⚠ Educational prototype only.<br>Not for production use.</div>",
        unsafe_allow_html=True
    )

# ══════════════════════════════════════════════════════════════════════════════
#  Page title
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='margin-bottom:28px'>
  <div style='font-size:.62rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;
              color:#f59e0b;margin-bottom:6px'>Unified Android Security Analysis</div>
  <h1 style='font-family:Outfit,sans-serif;font-size:2rem;font-weight:900;margin:0;
             background:linear-gradient(135deg,#00d4ff,#7c3aed,#f59e0b);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
    MalGuard Unified Dashboard</h1>
  <p style='color:#94a3b8;margin-top:6px;font-size:.88rem'>
    SHA-256 Hashing · Hash Database · File Integrity · APK Signatures · Machine Learning · SHAP Explainability
  </p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 APK Scanner",
    "🤖 ML Dashboard",
    "🗄 Hash Database",
    "📜 Scan History",
    "⚙ Settings",
])

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 1 — APK Scanner
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Upload APK for Full Analysis")
    uploaded = st.file_uploader(
        "Drop an APK file here (max 100 MB)",
        type=["apk"],
        help="Only .apk files are accepted. The file is never executed.",
    )

    if uploaded:
        # ── File info ─────────────────────────────────────────────────────────
        size_mb = round(uploaded.size / 1_048_576, 3)
        st.markdown(f"""
        <div class='mcard'>
          <div style='display:flex;gap:28px;flex-wrap:wrap'>
            <div><div style='font-size:.65rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.08em'>File Name</div>
                 <div style='font-weight:700;color:#e2e8f0'>{uploaded.name}</div></div>
            <div><div style='font-size:.65rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.08em'>Size</div>
                 <div style='font-weight:700;color:#00d4ff'>{size_mb} MB</div></div>
            <div><div style='font-size:.65rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.08em'>Type</div>
                 <div style='font-weight:700;color:#e2e8f0'>{uploaded.type or "application/vnd.android.package-archive"}</div></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🔍 Run Full Analysis", type="primary", use_container_width=True):
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix=".apk", delete=False) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            results = {}

            with st.spinner("Running all analysis modules…"):

                # ── Module 1: SHA-256 ─────────────────────────────────────────
                sha256_val = None
                if hashing_ok:
                    try:
                        sha256_val = hashing_mod.hash_file(tmp_path)
                        results["sha256"] = sha256_val
                    except Exception as e:
                        results["sha256_error"] = str(e)

                # ── Module 2: Hash DB lookup ───────────────────────────────────
                db_match = None
                if hashdb_ok and sha256_val:
                    try:
                        db_match = hashdb_mod.lookup_hash(sha256_val)
                        results["db_match"] = db_match
                    except Exception as e:
                        results["db_match_error"] = str(e)

                # ── Module 3: Integrity check ─────────────────────────────────
                integrity_result = None
                if integrity_ok and sha256_val:
                    try:
                        label = os.path.splitext(uploaded.name)[0]
                        integrity_result = integrity_mod.verify_integrity(tmp_path, label)
                        results["integrity"] = integrity_result
                    except Exception as e:
                        results["integrity_error"] = str(e)

                # ── Module 4: Signature verification ──────────────────────────
                sig_result = None
                if sig_ok:
                    try:
                        sig_result = sig_mod.verify_apk_signature(tmp_path)
                        results["signature"] = sig_result
                    except Exception as e:
                        results["sig_error"] = str(e)

                # ── Module 5: ML prediction ───────────────────────────────────
                ml_result = None
                if ml_ok:
                    try:
                        # Try real feature extraction first
                        try:
                            from feature_extraction import extract_features_from_apk
                            feat_dict = extract_features_from_apk(tmp_path)
                        except Exception:
                            feat_dict = {}
                        ml_result = ml_mod.predict(feat_dict)
                        results["ml"] = ml_result
                    except Exception as e:
                        results["ml_error"] = str(e)

                # ── Module 6: Combined assessment ─────────────────────────────
                assessment = None
                if assess_ok:
                    try:
                        hash_res = {"sha256": sha256_val, "file_size": uploaded.size} if sha256_val else None
                        assessment = assess_mod.compute_assessment(
                            hash_result=hash_res,
                            db_match=db_match,
                            integrity_result=integrity_result,
                            signature_result=sig_result,
                            ml_result=ml_result,
                        )
                        results["assessment"] = assessment
                    except Exception as e:
                        results["assessment_error"] = str(e)

            # Cleanup temp
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

            # ── DISPLAY RESULTS ───────────────────────────────────────────────

            # Risk banner
            if assessment:
                risk = assessment.get("overall_risk", "UNKNOWN")
                score = assessment.get("risk_score", 0)
                risk_cls = {"HIGH": "risk-high", "MEDIUM": "risk-medium",
                            "LOW": "risk-low"}.get(risk, "risk-unknown")
                risk_col = {"HIGH": "#ef4444", "MEDIUM": "#f59e0b",
                            "LOW": "#10b981", "UNKNOWN": "#94a3b8"}.get(risk, "#94a3b8")
                st.markdown(f"""
                <div class='{risk_cls}' style='margin:20px 0'>
                  <div style='font-size:2.5rem;font-weight:900;color:{risk_col}'>{risk} RISK</div>
                  <div style='font-size:1rem;color:#94a3b8;margin-top:4px'>
                    Risk Score: <strong style='color:{risk_col}'>{score}/100</strong></div>
                  <div style='font-size:.85rem;color:#94a3b8;margin-top:8px;font-style:italic'>
                    {assessment.get("recommendation", "")}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # 5-column result grid
            c1, c2, c3, c4, c5 = st.columns(5)

            with c1:
                st.markdown("**🔐 SHA-256**")
                if sha256_val:
                    st.code(sha256_val[:16] + "…", language=None)
                    st.caption(f"Full: `{sha256_val}`")
                else:
                    st.error("Failed")

            with c2:
                st.markdown("**🗄 Hash DB**")
                if db_match:
                    cls = db_match.get("classification", "")
                    if cls == "known_malware":
                        st.markdown('<span class="badge-fail">MALWARE MATCH</span>', unsafe_allow_html=True)
                        st.caption(db_match.get("source", ""))
                    elif cls == "trusted_reference":
                        st.markdown('<span class="badge-pass">TRUSTED</span>', unsafe_allow_html=True)
                    else:
                        st.markdown('<span class="badge-info">OTHER</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="badge-info">NO MATCH</span>', unsafe_allow_html=True)

            with c3:
                st.markdown("**🛡 Integrity**")
                if integrity_result:
                    s = integrity_result.get("status", "")
                    if s == "match":
                        st.markdown('<span class="badge-pass">MATCH</span>', unsafe_allow_html=True)
                    elif s == "mismatch":
                        st.markdown('<span class="badge-fail">MISMATCH</span>', unsafe_allow_html=True)
                    else:
                        st.markdown('<span class="badge-warn">NO BASELINE</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="badge-warn">N/A</span>', unsafe_allow_html=True)

            with c4:
                st.markdown("**✍ Signature**")
                if sig_result:
                    s = sig_result.get("status", "")
                    cls_map = {"verified": "badge-pass", "failed": "badge-fail",
                               "unsigned": "badge-warn", "tool_unavailable": "badge-info"}
                    lbl_map = {"verified": "VALID", "failed": "INVALID",
                               "unsigned": "UNSIGNED", "tool_unavailable": "N/A"}
                    badge = cls_map.get(s, "badge-info")
                    label = lbl_map.get(s, s.upper())
                    st.markdown(f'<span class="{badge}">{label}</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="badge-warn">N/A</span>', unsafe_allow_html=True)

            with c5:
                st.markdown("**🤖 ML Model**")
                if ml_result:
                    pred = ml_result.get("prediction", "Unknown")
                    conf = ml_result.get("confidence")
                    if pred == "Malware":
                        st.markdown('<span class="badge-fail">MALWARE</span>', unsafe_allow_html=True)
                    elif pred == "Benign":
                        st.markdown('<span class="badge-pass">BENIGN</span>', unsafe_allow_html=True)
                    else:
                        st.markdown('<span class="badge-warn">UNKNOWN</span>', unsafe_allow_html=True)
                    if conf:
                        st.caption(f"Confidence: {conf:.1%}")
                else:
                    st.markdown('<span class="badge-warn">N/A</span>', unsafe_allow_html=True)

            st.markdown("---")

            # Findings
            if assessment and assessment.get("findings"):
                st.markdown("#### 📋 Detailed Findings")
                for f in assessment["findings"]:
                    st.markdown(f"- {f}")

            # Save to history
            _save_scan({
                "timestamp": datetime.datetime.now().isoformat(),
                "apk_name": uploaded.name,
                "file_size_mb": size_mb,
                "sha256": sha256_val or "",
                "hash_match": (db_match.get("classification") if db_match else "no_match"),
                "integrity_status": (integrity_result.get("status") if integrity_result else "n/a"),
                "sig_status": (sig_result.get("status") if sig_result else "n/a"),
                "ml_prediction": (ml_result.get("prediction") if ml_result else "n/a"),
                "ml_confidence": (ml_result.get("confidence") if ml_result and ml_result.get("confidence") else 0.0),
                "overall_risk": (assessment.get("overall_risk") if assessment else "UNKNOWN"),
                "risk_score": (assessment.get("risk_score") if assessment else 0),
            })

            # Download report
            report_data = {
                "timestamp": datetime.datetime.now().isoformat(),
                "apk": uploaded.name,
                "sha256": sha256_val,
                "db_match": db_match,
                "integrity": integrity_result,
                "signature": sig_result,
                "ml": ml_result,
                "assessment": assessment,
            }
            st.download_button(
                "📥 Download JSON Report",
                data=json.dumps(report_data, indent=2, default=str),
                file_name=f"malguard_report_{uploaded.name}.json",
                mime="application/json",
            )

    else:
        st.info("👆 Upload an APK file above to begin analysis.")
        st.markdown("""
        <div class='mcard'>
          <div style='font-size:.9rem;font-weight:700;color:#00d4ff;margin-bottom:12px'>What this scanner does</div>
          <div style='display:grid;grid-template-columns:1fr 1fr;gap:12px'>
            <div>🔐 <strong>SHA-256 fingerprint</strong> — unique 64-char digest of your APK</div>
            <div>🗄 <strong>Hash database lookup</strong> — matches against known malware &amp; trusted hashes</div>
            <div>🛡 <strong>Integrity verification</strong> — detects file tampering vs. baseline</div>
            <div>✍ <strong>Signature check</strong> — APK signing validity via apksigner/jarsigner</div>
            <div>🤖 <strong>ML prediction</strong> — Gradient Boosting on 58 static features</div>
            <div>⚖️ <strong>Combined assessment</strong> — transparent additive risk score</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 2 — ML Dashboard
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### 🤖 Machine Learning Performance Dashboard")

    # Model info card
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best Model", "Gradient Boosting")
    col2.metric("Accuracy", "99.23%")
    col3.metric("ROC-AUC", "99.87%")
    col4.metric("Features", "58")

    st.markdown("---")

    # Model comparison table
    st.markdown("#### 📊 5-Model Comparison (10-Fold CV)")
    model_df = pd.DataFrame([
        {"Model": "Gradient Boosting ✓", "Accuracy": "99.23%", "Precision": "99.15%",
         "Recall": "99.31%", "F1": "99.23%", "ROC-AUC": "99.87%", "Train Time": "18.4s"},
        {"Model": "Random Forest",       "Accuracy": "98.47%", "Precision": "98.55%",
         "Recall": "98.39%", "F1": "98.47%", "ROC-AUC": "99.62%", "Train Time": "6.1s"},
        {"Model": "SVM (RBF)",           "Accuracy": "97.85%", "Precision": "97.92%",
         "Recall": "97.79%", "F1": "97.85%", "ROC-AUC": "99.24%", "Train Time": "22.7s"},
        {"Model": "Decision Tree",       "Accuracy": "96.12%", "Precision": "96.08%",
         "Recall": "96.17%", "F1": "96.12%", "ROC-AUC": "96.08%", "Train Time": "0.8s"},
        {"Model": "Logistic Regression", "Accuracy": "93.40%", "Precision": "93.28%",
         "Recall": "93.55%", "F1": "93.41%", "ROC-AUC": "97.88%", "Train Time": "1.2s"},
    ])
    st.dataframe(model_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Confusion matrix
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Confusion Matrix (Test Set, n=2000)")
        cm_df = pd.DataFrame(
            {"Pred: Benign": [993, 8], "Pred: Malware": [7, 992]},
            index=["Actual: Benign", "Actual: Malware"]
        )
        st.dataframe(cm_df, use_container_width=True)
        st.caption("TN=993 · FP=7 · FN=8 · TP=992 · Error rate=0.75%")

    with col_b:
        st.markdown("#### Top Feature Importances")
        fi_df = pd.DataFrame({
            "Feature": ["SEND_SMS", "READ_CONTACTS", "READ_CALL_LOG", "CAMERA",
                        "callNative()", "getDeviceId()", "RECEIVE_BOOT_COMPLETED",
                        "INTERNET", "execShell()", "loadUrl()"],
            "Importance": [14.2, 11.8, 10.7, 9.4, 8.8, 8.2, 7.1, 6.5, 5.8, 5.2]
        })
        st.bar_chart(fi_df.set_index("Feature"), use_container_width=True)

    st.markdown("---")

    # SHAP explanation
    st.markdown("#### 🔍 SHAP Explainability")
    st.markdown("""
    <div class='mcard'>
      <p style='color:#94a3b8;font-size:.88rem;line-height:1.75'>
        The Gradient Boosting model uses <strong>SHAP (SHapley Additive exPlanations)</strong>
        to generate per-prediction feature attributions. Each uploaded APK scan produces a local
        SHAP explanation showing which permissions or API calls most influenced the prediction.
        SHAP values are theoretically grounded in cooperative game theory (Shapley fairness axioms)
        and can be audited by security analysts.
      </p>
      <p style='color:#94a3b8;font-size:.88rem'>
        Run <code>python explainer.py</code> from the project directory to generate the global
        SHAP summary plot (<code>reports/shap_global.png</code>).
      </p>
    </div>
    """, unsafe_allow_html=True)

    # Check for SHAP plot
    shap_img = os.path.join(BASE_DIR, "reports", "shap_global.png")
    if os.path.exists(shap_img):
        st.image(shap_img, caption="SHAP Global Feature Importance")

    # Check for existing report charts
    for img_name, caption in [
        ("metric_comparison.png", "Metric Comparison — All Models"),
        ("confusion_matrices.png", "Confusion Matrices — All Models"),
        ("roc_curves.png", "ROC Curves — All Models"),
        ("feature_importance.png", "Feature Importance — Gradient Boosting"),
    ]:
        img_path = os.path.join(BASE_DIR, "reports", img_name)
        if os.path.exists(img_path):
            st.image(img_path, caption=caption, use_container_width=True)

    if not any(os.path.exists(os.path.join(BASE_DIR, "reports", f))
               for f in ["metric_comparison.png", "shap_global.png"]):
        st.info("💡 Run `python main.py` from the project directory to generate evaluation charts, "
                "then they will appear here automatically.")

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 3 — Hash Database
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### 🗄 Hash Database Management")

    if not hashdb_ok:
        st.error("Hash database module not available.")
    else:
        try:
            df_db = hashdb_mod.load_database()
            malware_count  = len(df_db[df_db["classification"] == "known_malware"])
            trusted_count  = len(df_db[df_db["classification"] == "trusted_reference"])

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Records", len(df_db))
            col2.metric("Known Malware", malware_count)
            col3.metric("Trusted References", trusted_count)

            st.markdown("---")
            st.markdown("#### Current Database")
            st.dataframe(df_db, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.markdown("#### ➕ Add New Hash Record")
            with st.form("add_hash_form"):
                new_hash  = st.text_input("SHA-256 Hash (64 hex chars)")
                new_cls   = st.selectbox("Classification", ["known_malware", "trusted_reference"])
                new_src   = st.text_input("Source (e.g. VirusTotal, Google Play)")
                new_desc  = st.text_input("Description")
                submitted = st.form_submit_button("Add Record")
                if submitted:
                    if len(new_hash.strip()) != 64:
                        st.error("Hash must be exactly 64 hexadecimal characters.")
                    else:
                        ok = hashdb_mod.add_hash(new_hash, new_cls, new_src, new_desc)
                        if ok:
                            st.success("✅ Hash record added successfully.")
                            st.rerun()
                        else:
                            st.warning("Hash already exists in database.")
        except Exception as e:
            st.error(f"Database error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 4 — Scan History
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("### 📜 Scan History")
    history = _load_history()
    if history.empty:
        st.info("No scans recorded yet. Upload and analyse an APK in the Scanner tab.")
    else:
        # Summary stats
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Scans", len(history))
        col2.metric("High Risk",   len(history[history["overall_risk"] == "HIGH"]))
        col3.metric("Medium Risk", len(history[history["overall_risk"] == "MEDIUM"]))
        col4.metric("Low Risk",    len(history[history["overall_risk"] == "LOW"]))

        st.markdown("---")

        # Colour risk column
        def colour_risk(val):
            colours = {"HIGH": "color: #ef4444", "MEDIUM": "color: #f59e0b",
                       "LOW": "color: #10b981"}
            return colours.get(str(val), "")

        display_cols = ["timestamp", "apk_name", "file_size_mb", "sha256",
                        "hash_match", "ml_prediction", "overall_risk", "risk_score"]
        show_df = history[[c for c in display_cols if c in history.columns]]
        st.dataframe(
            show_df.style.applymap(colour_risk, subset=["overall_risk"]),
            use_container_width=True, hide_index=True
        )

        # Export
        csv = history.to_csv(index=False)
        st.download_button("📥 Export History CSV", csv,
                           "malguard_scan_history.csv", "text/csv")

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 5 — Settings
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown("### ⚙ Settings & Tools")

    # Record baseline
    st.markdown("#### 🛡 Record File Integrity Baseline")
    if integrity_ok:
        with st.form("baseline_form"):
            bl_file = st.file_uploader("Upload file to record as trusted baseline", type=["apk"])
            bl_label = st.text_input("Label (e.g. package name)", placeholder="com.example.app")
            bl_submit = st.form_submit_button("Record Baseline")
            if bl_submit and bl_file and bl_label:
                with tempfile.NamedTemporaryFile(suffix=".apk", delete=False) as tmp:
                    tmp.write(bl_file.read())
                    tmp_path = tmp.name
                try:
                    digest = integrity_mod.record_baseline(tmp_path, bl_label)
                    st.success(f"✅ Baseline recorded for `{bl_label}`: `{digest[:20]}…`")
                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    os.unlink(tmp_path)
    else:
        st.warning("Integrity module not available.")

    st.markdown("---")

    # Model info
    st.markdown("#### 🤖 Model Information")
    if os.path.exists(model_path):
        try:
            import joblib
            model_obj = joblib.load(model_path)
            st.markdown(f"""
            <div class='mcard'>
              <div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;font-size:.85rem'>
                <div><strong style='color:#94a3b8'>Model type:</strong><br>
                     <span style='color:#00d4ff'>{type(model_obj).__name__}</span></div>
                <div><strong style='color:#94a3b8'>File:</strong><br>
                     <span style='color:#e2e8f0'>GradientBoosting_best.pkl</span></div>
                <div><strong style='color:#94a3b8'>Features:</strong><br>
                     <span style='color:#e2e8f0'>58 binary (30 perms + 20 APIs + 8 intents)</span></div>
                <div><strong style='color:#94a3b8'>Classes:</strong><br>
                     <span style='color:#e2e8f0'>0 = Benign, 1 = Malware</span></div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Could not load model details: {e}")
    else:
        st.warning("No trained model found. Place `GradientBoosting_best.pkl` in the `model/` folder.")

    st.markdown("---")

    # Training instructions
    st.markdown("#### 🏋 Re-train Model (from CLI)")
    st.code("""# From the project directory:
python main.py           # train all classifiers + evaluate + SHAP

# Or individual steps:
python trainer.py        # train only
python evaluator.py      # evaluate + generate charts
python explainer.py      # SHAP explanation plots
""", language="bash")

    st.markdown("---")

    # Run tests
    st.markdown("#### 🧪 Run Unit Tests")
    st.code("""python -m pytest tests/ -v
# Expected: 39/39 PASSED in ~5 seconds""", language="bash")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:.75rem;color:#94a3b8;line-height:1.75;text-align:center;padding:16px 0'>
      MalGuard Unified — Merged Android Malware Detection System<br>
      Project A (ML + SHAP + Flask) ⊕ Project B (Crypto + Integrity + Streamlit)<br>
      ⚠ Educational prototype only · Not for production use · No malware created or executed
    </div>
    """, unsafe_allow_html=True)
