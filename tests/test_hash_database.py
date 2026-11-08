"""
tests/test_hash_database.py — pytest tests for hash_database.py
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hash_database import (
    add_hash,
    create_empty_database,
    get_all_hashes,
    load_database,
    lookup_hash,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def db_path(tmp_path):
    """Return a fresh, pre-populated database path inside tmp_path."""
    path = str(tmp_path / "test_db.csv")
    # Trigger pre-population by loading (file doesn't exist yet)
    load_database(path)
    return path


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_lookup_hash_finds_known_malware(db_path):
    """lookup_hash should return a record for a pre-populated malware hash."""
    known_hash = "a3f1c2d4e5b6789012345678abcdef01234567890abcdef1234567890abcdef12"
    result = lookup_hash(known_hash, db_path)
    assert result is not None
    assert result["classification"] == "known_malware"


def test_lookup_hash_finds_trusted_reference(db_path):
    """lookup_hash should return a record for a pre-populated trusted hash."""
    trusted_hash = "0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20"
    result = lookup_hash(trusted_hash, db_path)
    assert result is not None
    assert result["classification"] == "trusted_reference"


def test_lookup_hash_returns_none_for_unknown(db_path):
    """lookup_hash should return None for a hash not in the database."""
    unknown_hash = "f" * 64
    result = lookup_hash(unknown_hash, db_path)
    assert result is None


def test_lookup_hash_case_insensitive(db_path):
    """lookup_hash should match regardless of hex case."""
    lower_hash = "a3f1c2d4e5b6789012345678abcdef01234567890abcdef1234567890abcdef12"
    upper_hash = lower_hash.upper()
    result = lookup_hash(upper_hash, db_path)
    assert result is not None


def test_add_hash_new_entry(db_path):
    """add_hash should return True when adding a genuinely new hash."""
    new_hash = "1234567890abcdef" * 4  # 64 chars
    ok = add_hash(
        sha256_hash=new_hash,
        classification="known_malware",
        source="test",
        description="test entry",
        db_path=db_path,
    )
    assert ok is True
    # Should be findable now
    result = lookup_hash(new_hash, db_path)
    assert result is not None


def test_add_hash_prevents_duplicates(db_path):
    """add_hash should return False when the hash already exists."""
    existing_hash = "a3f1c2d4e5b6789012345678abcdef01234567890abcdef1234567890abcdef12"
    ok = add_hash(
        sha256_hash=existing_hash,
        classification="known_malware",
        source="duplicate",
        description="should not be added",
        db_path=db_path,
    )
    assert ok is False


def test_add_hash_invalid_length(db_path):
    """add_hash should reject hashes that are not 64 hex characters."""
    ok = add_hash(
        sha256_hash="tooshort",
        classification="known_malware",
        source="test",
        description="bad hash",
        db_path=db_path,
    )
    assert ok is False


def test_get_all_hashes_returns_dataframe(db_path):
    """get_all_hashes should return a non-empty DataFrame."""
    df = get_all_hashes(db_path)
    assert len(df) > 0
    assert "sha256_hash" in df.columns
    assert "classification" in df.columns


def test_create_empty_database(tmp_path):
    """create_empty_database should create a CSV with only headers."""
    path = str(tmp_path / "empty.csv")
    create_empty_database(path)
    assert os.path.exists(path)
    with open(path) as f:
        content = f.read()
    assert "sha256_hash" in content


def test_load_database_creates_if_missing(tmp_path):
    """load_database should create and pre-populate if file is absent."""
    path = str(tmp_path / "new_db.csv")
    assert not os.path.exists(path)
    df = load_database(path)
    assert os.path.exists(path)
    assert len(df) >= 8  # 5 malware + 3 trusted
