"""
assessment.py — Combines all analysis module results into a final risk assessment.

The scoring rules are transparent and deliberately conservative:

  +80 pts  – Hash matched in database as *known_malware*
  +40 pts  – ML model predicts *Malware*
  +20 pts  – Integrity baseline *mismatch* (file was modified)
  +10 pts  – Digital signature verification *failed*

  Score is capped at 100.
  Risk level thresholds: HIGH ≥ 70, MEDIUM 40–69, LOW < 40.
  UNKNOWN is returned only when every module has explicitly reported a failure.

Usage::

    from assessment import compute_assessment

    result = compute_assessment(
        hash_result=..., db_match=..., integrity_result=...,
        signature_result=..., ml_result=...
    )
    print(result["overall_risk"])   # "HIGH" | "MEDIUM" | "LOW" | "UNKNOWN"
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def compute_assessment(
    hash_result: Optional[dict],
    db_match: Optional[dict],
    integrity_result: Optional[dict],
    signature_result: Optional[dict],
    ml_result: Optional[dict],
) -> dict:
    """
    Aggregate the outputs of all analysis modules into a single risk assessment.

    Parameters
    ----------
    hash_result : dict or None
        Output from the hashing step, e.g. ``{"sha256": "...", "file_size": 12345}``.
    db_match : dict or None
        Record returned by :func:`hash_database.lookup_hash`, or ``None`` if
        no match was found.
    integrity_result : dict or None
        Output of :func:`integrity_check.verify_integrity`.
    signature_result : dict or None
        Output of :func:`signature_check.verify_apk_signature`.
    ml_result : dict or None
        Output of :func:`malware_prediction.predict`.

    Returns
    -------
    dict
        Assessment dictionary with keys:

        ``overall_risk``
            ``"HIGH"``, ``"MEDIUM"``, ``"LOW"``, or ``"UNKNOWN"``.
        ``risk_score``
            Integer 0–100.
        ``findings``
            List of human-readable finding strings.
        ``recommendation``
            A single actionable recommendation string.
        ``details``
            Dict containing all five input dicts for full auditability.
    """
    risk_score = 0
    findings: list = []
    module_failures = 0
    total_modules = 5  # hash, db, integrity, signature, ml

    # ── Track whether inputs are available ────────────────────────────────────
    if hash_result is None:
        module_failures += 1
        findings.append("⚠ Hashing module did not return a result.")

    # ── Rule 1: Hash database match ───────────────────────────────────────────
    if db_match is None:
        findings.append("ℹ Hash database: no exact match found for this file.")
        if hash_result is None:
            module_failures += 1
    else:
        classification = db_match.get("classification", "").lower()
        source = db_match.get("source", "unknown source")
        description = db_match.get("description", "")

        if classification == "known_malware":
            risk_score += 80
            findings.append(
                f"🚨 Hash matched KNOWN MALWARE in database "
                f"(source: {source} — {description})."
            )
        elif classification == "trusted_reference":
            findings.append(
                f"✅ Hash matched TRUSTED REFERENCE in database "
                f"(source: {source} — {description})."
            )
        else:
            findings.append(
                f"ℹ Hash database match with unknown classification: '{classification}'."
            )

    # ── Rule 2: ML prediction ─────────────────────────────────────────────────
    if ml_result is None:
        module_failures += 1
        findings.append("⚠ ML prediction module did not return a result.")
    else:
        ml_error = ml_result.get("error")
        prediction = ml_result.get("prediction", "Unknown")
        confidence = ml_result.get("confidence")
        model_name = ml_result.get("model_name", "Unknown model")
        conf_str = f" (confidence: {confidence:.1%})" if confidence is not None else ""

        if prediction == "Malware":
            risk_score += 40
            findings.append(
                f"🤖 ML model ({model_name}) predicts MALWARE{conf_str}."
            )
        elif prediction == "Benign":
            findings.append(
                f"✅ ML model ({model_name}) predicts BENIGN{conf_str}."
            )
        else:
            module_failures += 1
            error_detail = f": {ml_error}" if ml_error else ""
            findings.append(
                f"⚠ ML prediction unavailable{error_detail}."
            )

    # ── Rule 3: Integrity verification ────────────────────────────────────────
    if integrity_result is None:
        module_failures += 1
        findings.append("⚠ Integrity check module did not return a result.")
    else:
        status = integrity_result.get("status", "")
        message = integrity_result.get("message", "")

        if status == "mismatch":
            risk_score += 20
            findings.append(f"⚠ Integrity check: MISMATCH — {message}")
        elif status == "match":
            findings.append(f"✅ Integrity check: file matches stored baseline.")
        elif status == "no_baseline":
            findings.append("ℹ Integrity check: no baseline recorded for this file.")
        else:
            module_failures += 1
            findings.append(f"⚠ Integrity check error: {message}")

    # ── Rule 4: Signature verification ───────────────────────────────────────
    if signature_result is None:
        module_failures += 1
        findings.append("⚠ Signature verification module did not return a result.")
    else:
        sig_status = signature_result.get("status", "")
        tool_used = signature_result.get("tool_used", "unknown")

        if sig_status == "failed":
            risk_score += 10
            findings.append(
                f"⚠ Signature verification FAILED (tool: {tool_used}). "
                "The APK may be tampered with or self-signed with an untrusted key."
            )
        elif sig_status == "verified":
            findings.append(
                f"✅ APK signature verified (tool: {tool_used})."
            )
        elif sig_status == "unsigned":
            risk_score += 10
            findings.append(
                "⚠ APK is UNSIGNED — legitimate apps published on app stores are always signed."
            )
        elif sig_status == "tool_unavailable":
            findings.append(
                "ℹ Signature verification tool not available on this system "
                "(install Android SDK Build-Tools or JDK)."
            )
        else:
            module_failures += 1
            findings.append(
                f"⚠ Signature verification error: {signature_result.get('message', '')}."
            )

    # ── Cap score ─────────────────────────────────────────────────────────────
    risk_score = min(risk_score, 100)

    # ── Determine overall risk level ──────────────────────────────────────────
    if module_failures >= total_modules:
        overall_risk = "UNKNOWN"
    elif risk_score >= 70:
        overall_risk = "HIGH"
    elif risk_score >= 40:
        overall_risk = "MEDIUM"
    else:
        overall_risk = "LOW"

    # ── Recommendation ────────────────────────────────────────────────────────
    recommendations = {
        "HIGH": (
            "This file shows strong indicators of malware. "
            "Do NOT install it. Quarantine and report it to your security team immediately."
        ),
        "MEDIUM": (
            "This file shows some suspicious characteristics. "
            "Exercise caution — verify its source before installation. "
            "Consider a deeper manual review."
        ),
        "LOW": (
            "No strong malware indicators were detected. "
            "This assessment is NOT a guarantee of safety. "
            "Only install APKs from trusted, verified sources."
        ),
        "UNKNOWN": (
            "Analysis could not be completed because one or more modules failed. "
            "Do not install this file until a full analysis is possible."
        ),
    }
    recommendation = recommendations[overall_risk]

    return {
        "overall_risk": overall_risk,
        "risk_score": risk_score,
        "findings": findings,
        "recommendation": recommendation,
        "details": {
            "hash_result": hash_result,
            "db_match": db_match,
            "integrity_result": integrity_result,
            "signature_result": signature_result,
            "ml_result": ml_result,
        },
    }
