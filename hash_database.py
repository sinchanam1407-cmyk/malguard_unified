"""
hash_database.py — CSV-backed SHA-256 hash database for known malware and
trusted reference files.

The database is stored as a CSV file with the following columns:
    id, sha256_hash, classification, source, date_added, description

``classification`` is one of:
    - ``known_malware``      – confirmed malicious APK
    - ``trusted_reference``  – verified benign / official APK

Usage:
    from hash_database import lookup_hash, add_hash, load_database
"""

import csv
import logging
import os
import uuid
from datetime import date
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Default database path (relative to this file)
_DEFAULT_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "hash_database.csv")

# Required columns
_COLUMNS = ["id", "sha256_hash", "classification", "source", "date_added", "description"]

# Pre-populated sample records
_SAMPLE_RECORDS = [
    # ── Known malware ──────────────────────────────────────────────────────────
    {
        "id": "1",
        "sha256_hash": "a3f1c2d4e5b6789012345678abcdef01234567890abcdef1234567890abcdef12",
        "classification": "known_malware",
        "source": "VirusTotal",
        "date_added": "2024-01-15",
        "description": "Banking trojan — steals OTP via SMS interception",
    },
    {
        "id": "2",
        "sha256_hash": "deadbeef0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c",
        "classification": "known_malware",
        "source": "MalwareBazaar",
        "date_added": "2024-03-22",
        "description": "SpyNote RAT — remote access trojan",
    },
    {
        "id": "3",
        "sha256_hash": "cafebabe1122334455667788990011223344556677889900112233445566778899",
        "classification": "known_malware",
        "source": "Hybrid-Analysis",
        "date_added": "2024-05-10",
        "description": "Joker spyware — subscribes user to premium SMS services",
    },
    {
        "id": "4",
        "sha256_hash": "feedface00112233445566778899aabbccddeeff00112233445566778899aabb",
        "classification": "known_malware",
        "source": "CERT-In",
        "date_added": "2024-07-03",
        "description": "Anubis banking malware — credential harvester",
    },
    {
        "id": "5",
        "sha256_hash": "baadf00d1234567890abcdef1234567890abcdef1234567890abcdef12345678",
        "classification": "known_malware",
        "source": "VirusTotal",
        "date_added": "2024-09-18",
        "description": "FluBot SMS worm — spreads via fake delivery notifications",
    },
    # ── Trusted references ──────────────────────────────────────────────────────
    {
        "id": "6",
        "sha256_hash": "0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20",
        "classification": "trusted_reference",
        "source": "Google Play",
        "date_added": "2024-02-01",
        "description": "Official Google Chrome APK v121.0.6167.143",
    },
    {
        "id": "7",
        "sha256_hash": "f0e1d2c3b4a59687786960504030201f0e1d2c3b4a596877869605040302010",
        "classification": "trusted_reference",
        "source": "Google Play",
        "date_added": "2024-04-15",
        "description": "Official WhatsApp Messenger APK v2.24.3.77",
    },
    {
        "id": "8",
        "sha256_hash": "1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b",
        "classification": "trusted_reference",
        "source": "AOSP",
        "date_added": "2024-06-20",
        "description": "Android Settings APK from AOSP 14 release build",
    },
]


def create_empty_database(db_path: str = _DEFAULT_DB) -> None:
    """
    Create an empty hash database CSV with the required headers.

    If the file already exists this function is a no-op.

    Parameters
    ----------
    db_path : str
        Path where the CSV should be created (parent directories are made).
    """
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    if os.path.exists(db_path):
        return
    with open(db_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_COLUMNS)
        writer.writeheader()
    logger.info("Created empty hash database at %s", db_path)


def _init_database(db_path: str) -> None:
    """Create the database and pre-populate it with sample records."""
    create_empty_database(db_path)
    df = pd.read_csv(db_path)
    if len(df) == 0:
        sample_df = pd.DataFrame(_SAMPLE_RECORDS, columns=_COLUMNS)
        sample_df.to_csv(db_path, index=False)
        logger.info("Pre-populated hash database with %d sample records.", len(_SAMPLE_RECORDS))


def load_database(db_path: str = _DEFAULT_DB) -> pd.DataFrame:
    """
    Load the hash database from a CSV file.

    If the file does not exist it is created and pre-populated with sample
    records before being returned.

    Parameters
    ----------
    db_path : str
        Path to the CSV database file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: id, sha256_hash, classification, source,
        date_added, description.
    """
    if not os.path.exists(db_path):
        _init_database(db_path)
    try:
        df = pd.read_csv(db_path, dtype=str).fillna("")
        if df.empty:
            _init_database(db_path)
            df = pd.read_csv(db_path, dtype=str).fillna("")
        # Normalise hashes to lowercase
        df["sha256_hash"] = df["sha256_hash"].str.lower()
        return df
    except Exception as exc:
        logger.error("load_database: failed to read '%s': %s", db_path, exc)
        return pd.DataFrame(columns=_COLUMNS)


def lookup_hash(hash_value: str, db_path: str = _DEFAULT_DB) -> Optional[dict]:
    """
    Look up a SHA-256 hash in the database.

    Parameters
    ----------
    hash_value : str
        The hex SHA-256 digest to search for.
    db_path : str
        Path to the CSV database file.

    Returns
    -------
    dict or None
        The matching record as a plain dict, or ``None`` if not found.
    """
    normalised = hash_value.lower().strip()
    df = load_database(db_path)
    matches = df[df["sha256_hash"] == normalised]
    if matches.empty:
        return None
    return matches.iloc[0].to_dict()


def add_hash(
    sha256_hash: str,
    classification: str,
    source: str,
    description: str,
    db_path: str = _DEFAULT_DB,
) -> bool:
    """
    Add a new hash record to the database.

    Duplicate hashes (case-insensitive) are silently rejected.

    Parameters
    ----------
    sha256_hash : str
        64-character hex SHA-256 digest.
    classification : str
        ``"known_malware"`` or ``"trusted_reference"``.
    source : str
        Where this hash was obtained from.
    description : str
        Human-readable description of the file.
    db_path : str
        Path to the CSV database file.

    Returns
    -------
    bool
        ``True`` if the record was added; ``False`` if it already existed or
        an error occurred.
    """
    normalised = sha256_hash.lower().strip()
    if len(normalised) != 64:
        logger.warning("add_hash: '%s' is not a valid 64-char SHA-256 hex string.", normalised)
        return False

    df = load_database(db_path)
    if not df[df["sha256_hash"] == normalised].empty:
        logger.info("add_hash: hash already exists in database — skipping.")
        return False

    new_id = str(int(df["id"].apply(lambda x: int(x) if str(x).isdigit() else 0).max() or 0) + 1)
    new_record = {
        "id": new_id,
        "sha256_hash": normalised,
        "classification": classification,
        "source": source,
        "date_added": date.today().isoformat(),
        "description": description,
    }
    new_row = pd.DataFrame([new_record], columns=_COLUMNS)
    updated = pd.concat([df, new_row], ignore_index=True)
    try:
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        updated.to_csv(db_path, index=False)
        logger.info("add_hash: added new record id=%s", new_id)
        return True
    except OSError as exc:
        logger.error("add_hash: could not write database: %s", exc)
        return False


def get_all_hashes(db_path: str = _DEFAULT_DB) -> pd.DataFrame:
    """
    Return the full hash database as a DataFrame.

    Parameters
    ----------
    db_path : str
        Path to the CSV database file.

    Returns
    -------
    pd.DataFrame
        All records in the database.
    """
    return load_database(db_path)
