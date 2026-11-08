"""
tests/test_assessment.py — pytest tests for assessment.py
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from assessment import compute_assessment


# ── Fixtures / helpers ────────────────────────────────────────────────────────

def _hash_result(sha256: str = "a" * 64) -> dict:
    return {"sha256": sha256, "file_size": 1000}


def _no_match() -> None:
    return None


def _known_malware_match() -> dict:
    return {
        "id": "1",
        "sha256_hash": "a" * 64,
        "classification": "known_malware",
        "source": "VirusTotal",
        "date_added": "2024-01-01",
        "description": "Test malware sample",
    }


def _trusted_match() -> dict:
    return {
        "id": "6",
        "sha256_hash": "b" * 64,
        "classification": "trusted_reference",
        "source": "Google Play",
        "date_added": "2024-01-01",
        "description": "Official app",
    }


def _integrity_match() -> dict:
    return {
        "status": "match",
        "stored_hash": "a" * 64,
        "current_hash": "a" * 64,
        "message": "File matches baseline.",
    }


def _integrity_mismatch() -> dict:
    return {
        "status": "mismatch",
        "stored_hash": "a" * 64,
        "current_hash": "b" * 64,
        "message": "Hash mismatch.",
    }


def _no_baseline() -> dict:
    return {
        "status": "no_baseline",
        "stored_hash": None,
        "current_hash": None,
        "message": "No baseline recorded.",
    }


def _sig_verified() -> dict:
    return {
        "status": "verified",
        "tool_used": "apksigner",
        "certificate_info": "Cert info here",
        "message": "Verified.",
    }


def _sig_failed() -> dict:
    return {
        "status": "failed",
        "tool_used": "apksigner",
        "certificate_info": "",
        "message": "Verification failed.",
    }


def _sig_unavailable() -> dict:
    return {
        "status": "tool_unavailable",
        "tool_used": "none",
        "certificate_info": "",
        "message": "No tool found.",
    }


def _ml_malware() -> dict:
    return {
        "prediction": "Malware",
        "confidence": 0.92,
        "model_name": "GradientBoostingClassifier",
        "feature_count": 58,
        "error": None,
    }


def _ml_benign() -> dict:
    return {
        "prediction": "Benign",
        "confidence": 0.88,
        "model_name": "GradientBoostingClassifier",
        "feature_count": 58,
        "error": None,
    }


def _ml_unknown() -> dict:
    return {
        "prediction": "Unknown",
        "confidence": None,
        "model_name": "N/A",
        "feature_count": 58,
        "error": "Model not found.",
    }


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_high_risk_on_known_malware_match():
    """Known malware hash match alone should push risk to HIGH."""
    result = compute_assessment(
        hash_result=_hash_result(),
        db_match=_known_malware_match(),
        integrity_result=_no_baseline(),
        signature_result=_sig_unavailable(),
        ml_result=_ml_benign(),
    )
    assert result["overall_risk"] == "HIGH"
    assert result["risk_score"] >= 70


def test_high_risk_on_malware_match_plus_ml():
    """Combined known malware DB match + ML malware → score 80+40=120 → capped at 100."""
    result = compute_assessment(
        hash_result=_hash_result(),
        db_match=_known_malware_match(),
        integrity_result=_no_baseline(),
        signature_result=_sig_unavailable(),
        ml_result=_ml_malware(),
    )
    assert result["overall_risk"] == "HIGH"
    assert result["risk_score"] == 100  # capped


def test_medium_risk_ml_malware_only():
    """ML predicts malware but no DB match → score=40 → MEDIUM."""
    result = compute_assessment(
        hash_result=_hash_result(),
        db_match=None,
        integrity_result=_no_baseline(),
        signature_result=_sig_unavailable(),
        ml_result=_ml_malware(),
    )
    assert result["overall_risk"] == "MEDIUM"
    assert result["risk_score"] == 40


def test_low_risk_all_clear():
    """No malicious indicators → LOW risk."""
    result = compute_assessment(
        hash_result=_hash_result(),
        db_match=None,
        integrity_result=_integrity_match(),
        signature_result=_sig_verified(),
        ml_result=_ml_benign(),
    )
    assert result["overall_risk"] == "LOW"
    assert result["risk_score"] == 0


def test_unknown_when_all_modules_fail():
    """When every module returns None the risk should be UNKNOWN."""
    result = compute_assessment(
        hash_result=None,
        db_match=None,
        integrity_result=None,
        signature_result=None,
        ml_result=None,
    )
    assert result["overall_risk"] == "UNKNOWN"


def test_risk_score_cap_at_100():
    """Risk score must never exceed 100."""
    result = compute_assessment(
        hash_result=_hash_result(),
        db_match=_known_malware_match(),   # +80
        integrity_result=_integrity_mismatch(),  # +20
        signature_result=_sig_failed(),          # +10
        ml_result=_ml_malware(),                 # +40  → total 150 → capped 100
    )
    assert result["risk_score"] <= 100


def test_integrity_mismatch_contributes_to_score():
    """Integrity mismatch should add 20 to the risk score."""
    result_no_mismatch = compute_assessment(
        hash_result=_hash_result(),
        db_match=None,
        integrity_result=_no_baseline(),
        signature_result=_sig_unavailable(),
        ml_result=_ml_benign(),
    )
    result_with_mismatch = compute_assessment(
        hash_result=_hash_result(),
        db_match=None,
        integrity_result=_integrity_mismatch(),
        signature_result=_sig_unavailable(),
        ml_result=_ml_benign(),
    )
    assert result_with_mismatch["risk_score"] == result_no_mismatch["risk_score"] + 20


def test_signature_failed_contributes_to_score():
    """Failed signature verification should add 10 to the risk score."""
    base = compute_assessment(
        hash_result=_hash_result(),
        db_match=None,
        integrity_result=_no_baseline(),
        signature_result=_sig_unavailable(),
        ml_result=_ml_benign(),
    )
    with_sig_fail = compute_assessment(
        hash_result=_hash_result(),
        db_match=None,
        integrity_result=_no_baseline(),
        signature_result=_sig_failed(),
        ml_result=_ml_benign(),
    )
    assert with_sig_fail["risk_score"] == base["risk_score"] + 10


def test_trusted_reference_match_low_risk():
    """A trusted reference match should keep risk LOW."""
    result = compute_assessment(
        hash_result=_hash_result(),
        db_match=_trusted_match(),
        integrity_result=_integrity_match(),
        signature_result=_sig_verified(),
        ml_result=_ml_benign(),
    )
    assert result["overall_risk"] == "LOW"


def test_findings_list_is_populated():
    """The findings list should always contain at least one entry."""
    result = compute_assessment(
        hash_result=_hash_result(),
        db_match=None,
        integrity_result=_no_baseline(),
        signature_result=_sig_unavailable(),
        ml_result=_ml_benign(),
    )
    assert isinstance(result["findings"], list)
    assert len(result["findings"]) > 0


def test_details_contains_all_inputs():
    """The details sub-dict should contain all five input components."""
    hr = _hash_result()
    dm = _known_malware_match()
    ir = _no_baseline()
    sr = _sig_unavailable()
    mr = _ml_malware()
    result = compute_assessment(
        hash_result=hr,
        db_match=dm,
        integrity_result=ir,
        signature_result=sr,
        ml_result=mr,
    )
    assert result["details"]["hash_result"] == hr
    assert result["details"]["db_match"] == dm
    assert result["details"]["integrity_result"] == ir
    assert result["details"]["signature_result"] == sr
    assert result["details"]["ml_result"] == mr
