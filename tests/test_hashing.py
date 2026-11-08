"""
tests/test_hashing.py — pytest tests for hashing.py
"""

import hashlib
import os
import sys

import pytest

# Make sure the project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hashing import hash_bytes, hash_file, verify_hash


# ── Test 1: hash_bytes returns a valid 64-char hex string ─────────────────────
def test_hash_bytes_returns_64_char_hex():
    digest = hash_bytes(b"hello world")
    assert isinstance(digest, str)
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


# ── Test 2: same content → same hash ─────────────────────────────────────────
def test_hash_bytes_deterministic():
    data = b"android malware detection"
    assert hash_bytes(data) == hash_bytes(data)


# ── Test 3: different content → different hash ────────────────────────────────
def test_hash_bytes_different_inputs_differ():
    assert hash_bytes(b"good app") != hash_bytes(b"bad app")


# ── Test 4: verify_hash returns True for correct hash ─────────────────────────
def test_verify_hash_correct(tmp_path):
    f = tmp_path / "sample.apk"
    content = b"fake apk content for testing"
    f.write_bytes(content)
    expected = hashlib.sha256(content).hexdigest()
    assert verify_hash(str(f), expected) is True


# ── Test 5: verify_hash returns False for wrong hash ─────────────────────────
def test_verify_hash_wrong(tmp_path):
    f = tmp_path / "sample.apk"
    f.write_bytes(b"some content")
    wrong_hash = "a" * 64
    assert verify_hash(str(f), wrong_hash) is False


# ── Test 6: verify_hash is case-insensitive ───────────────────────────────────
def test_verify_hash_case_insensitive(tmp_path):
    f = tmp_path / "sample.bin"
    content = b"case insensitive test"
    f.write_bytes(content)
    lower_hash = hashlib.sha256(content).hexdigest()
    upper_hash = lower_hash.upper()
    assert verify_hash(str(f), upper_hash) is True


# ── Test 7: hash_file works on a temp file ───────────────────────────────────
def test_hash_file_on_temp_file(tmp_path):
    f = tmp_path / "testfile.bin"
    content = b"chunked reading test " * 500  # > 8192 bytes
    f.write_bytes(content)
    expected = hashlib.sha256(content).hexdigest()
    assert hash_file(str(f)) == expected


# ── Test 8: hash_file raises FileNotFoundError for missing file ───────────────
def test_hash_file_missing_raises():
    with pytest.raises(FileNotFoundError):
        hash_file("/nonexistent/path/to/file.apk")


# ── Test 9: hash_bytes of empty bytes is consistent with hashlib ──────────────
def test_hash_bytes_empty():
    assert hash_bytes(b"") == hashlib.sha256(b"").hexdigest()


# ── Test 10: verify_hash returns False for missing file ───────────────────────
def test_verify_hash_missing_file():
    assert verify_hash("/no/such/file.apk", "a" * 64) is False
