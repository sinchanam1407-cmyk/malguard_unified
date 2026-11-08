"""
tests/test_integrity.py — pytest tests for integrity_check.py
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from integrity_check import list_baselines, record_baseline, remove_baseline, verify_integrity


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def baselines_file(tmp_path):
    """Return a path to a fresh (non-existent) baselines JSON file."""
    return str(tmp_path / "baselines.json")


@pytest.fixture()
def sample_file(tmp_path):
    """Create a small file to use as the target in tests."""
    f = tmp_path / "app.apk"
    f.write_bytes(b"PK\x03\x04" + b"fake apk content for integrity testing")
    return str(f)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_record_baseline_records_hash(sample_file, baselines_file):
    """record_baseline should return a 64-char hex string and save it."""
    digest = record_baseline(sample_file, label="test_app", baselines_file=baselines_file)
    assert isinstance(digest, str)
    assert len(digest) == 64
    # Verify the file was written
    with open(baselines_file) as f:
        data = json.load(f)
    assert "test_app" in data
    assert data["test_app"]["hash"] == digest


def test_verify_integrity_match(sample_file, baselines_file):
    """verify_integrity should return 'match' for an unchanged file."""
    record_baseline(sample_file, label="test_app", baselines_file=baselines_file)
    result = verify_integrity(sample_file, label="test_app", baselines_file=baselines_file)
    assert result["status"] == "match"
    assert result["stored_hash"] == result["current_hash"]


def test_verify_integrity_mismatch(sample_file, baselines_file, tmp_path):
    """verify_integrity should return 'mismatch' when the file is modified."""
    record_baseline(sample_file, label="test_app", baselines_file=baselines_file)
    # Modify the file
    with open(sample_file, "ab") as f:
        f.write(b"\x00 TAMPERED")
    result = verify_integrity(sample_file, label="test_app", baselines_file=baselines_file)
    assert result["status"] == "mismatch"
    assert result["stored_hash"] != result["current_hash"]


def test_verify_integrity_no_baseline(sample_file, baselines_file):
    """verify_integrity should return 'no_baseline' for an unrecorded file."""
    result = verify_integrity(sample_file, label="unrecorded_app", baselines_file=baselines_file)
    assert result["status"] == "no_baseline"
    assert result["stored_hash"] is None


def test_verify_integrity_missing_file(tmp_path, baselines_file):
    """verify_integrity should handle a missing file gracefully."""
    # First record with the file present
    f = tmp_path / "gone.apk"
    f.write_bytes(b"content")
    record_baseline(str(f), label="gone_app", baselines_file=baselines_file)
    # Remove the file
    f.unlink()
    result = verify_integrity(str(f), label="gone_app", baselines_file=baselines_file)
    # Should report an error, not raise
    assert result["status"] in ("error", "mismatch")


def test_list_baselines_returns_dict(sample_file, baselines_file):
    """list_baselines should return the recorded baselines as a dict."""
    record_baseline(sample_file, label="app_a", baselines_file=baselines_file)
    baselines = list_baselines(baselines_file)
    assert isinstance(baselines, dict)
    assert "app_a" in baselines


def test_remove_baseline_existing(sample_file, baselines_file):
    """remove_baseline should return True and delete the entry."""
    record_baseline(sample_file, label="removable", baselines_file=baselines_file)
    assert remove_baseline("removable", baselines_file=baselines_file) is True
    baselines = list_baselines(baselines_file)
    assert "removable" not in baselines


def test_remove_baseline_nonexistent(baselines_file):
    """remove_baseline should return False for a label that doesn't exist."""
    assert remove_baseline("ghost", baselines_file=baselines_file) is False
