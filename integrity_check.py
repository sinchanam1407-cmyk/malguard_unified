"""
integrity_check.py — Trusted file baseline management using SHA-256 hashes.

Baselines are stored as a JSON file (default: data/trusted_baselines.json) with
the structure::

    {
        "<label>": {
            "hash": "<sha256_hex>",
            "date_recorded": "<ISO-8601 date>",
            "source": "<description of origin>"
        },
        ...
    }

Usage::

    from integrity_check import record_baseline, verify_integrity

    record_baseline("app.apk", label="my_app", baselines_file="data/trusted_baselines.json")
    result = verify_integrity("app.apk", label="my_app", baselines_file="data/trusted_baselines.json")
"""

import json
import logging
import os
from datetime import date
from typing import Any

from hashing import hash_file

logger = logging.getLogger(__name__)

# Default baselines file path
_DEFAULT_BASELINES = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "trusted_baselines.json"
)


def _load_baselines(baselines_file: str) -> dict:
    """Load the baselines JSON, returning an empty dict on any error."""
    if not os.path.exists(baselines_file):
        return {}
    try:
        with open(baselines_file, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("_load_baselines: could not read '%s': %s", baselines_file, exc)
        return {}


def _save_baselines(data: dict, baselines_file: str) -> None:
    """Persist the baselines dict back to disk."""
    os.makedirs(os.path.dirname(os.path.abspath(baselines_file)), exist_ok=True)
    with open(baselines_file, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def record_baseline(
    file_path: str,
    label: str,
    baselines_file: str = _DEFAULT_BASELINES,
) -> str:
    """
    Compute the SHA-256 hash of *file_path* and store it as a trusted baseline
    under *label*.

    If a baseline for *label* already exists it is overwritten.

    Parameters
    ----------
    file_path : str
        Path to the file to record.
    label : str
        Logical name used to identify this baseline (e.g., the APK package name).
    baselines_file : str
        Path to the JSON file where baselines are stored.

    Returns
    -------
    str
        The SHA-256 hex digest that was recorded.

    Raises
    ------
    FileNotFoundError
        If *file_path* does not exist.
    OSError
        If the file cannot be read or the baselines file cannot be written.
    """
    digest = hash_file(file_path)
    baselines = _load_baselines(baselines_file)
    baselines[label] = {
        "hash": digest,
        "date_recorded": date.today().isoformat(),
        "source": os.path.basename(file_path),
    }
    _save_baselines(baselines, baselines_file)
    logger.info("record_baseline: recorded '%s' → %s", label, digest)
    return digest


def verify_integrity(
    file_path: str,
    label: str,
    baselines_file: str = _DEFAULT_BASELINES,
) -> dict:
    """
    Verify whether *file_path* matches the stored baseline for *label*.

    Parameters
    ----------
    file_path : str
        Path to the file to check.
    label : str
        The logical name whose baseline should be compared.
    baselines_file : str
        Path to the JSON baselines file.

    Returns
    -------
    dict
        A result dictionary with keys:

        ``status``
            One of ``"match"``, ``"mismatch"``, or ``"no_baseline"``.
        ``stored_hash``
            The baseline digest (or ``None`` if no baseline exists).
        ``current_hash``
            The freshly computed digest (or ``None`` on I/O error).
        ``message``
            Human-readable description of the outcome.
    """
    baselines = _load_baselines(baselines_file)

    if label not in baselines:
        return {
            "status": "no_baseline",
            "stored_hash": None,
            "current_hash": None,
            "message": f"No baseline recorded for label '{label}'.",
        }

    stored_hash: str = baselines[label].get("hash", "")

    try:
        current_hash = hash_file(file_path)
    except (FileNotFoundError, OSError) as exc:
        return {
            "status": "error",
            "stored_hash": stored_hash,
            "current_hash": None,
            "message": f"Could not read file '{file_path}': {exc}",
        }

    if current_hash.lower() == stored_hash.lower():
        return {
            "status": "match",
            "stored_hash": stored_hash,
            "current_hash": current_hash,
            "message": "File integrity verified — hash matches the stored baseline.",
        }
    else:
        return {
            "status": "mismatch",
            "stored_hash": stored_hash,
            "current_hash": current_hash,
            "message": "WARNING: Hash mismatch — the file may have been tampered with.",
        }


def list_baselines(baselines_file: str = _DEFAULT_BASELINES) -> dict:
    """
    Return all stored baselines.

    Parameters
    ----------
    baselines_file : str
        Path to the JSON baselines file.

    Returns
    -------
    dict
        Mapping of label → baseline record.
    """
    return _load_baselines(baselines_file)


def remove_baseline(label: str, baselines_file: str = _DEFAULT_BASELINES) -> bool:
    """
    Delete the baseline entry for *label*.

    Parameters
    ----------
    label : str
        The label to remove.
    baselines_file : str
        Path to the JSON baselines file.

    Returns
    -------
    bool
        ``True`` if the label existed and was removed; ``False`` otherwise.
    """
    baselines = _load_baselines(baselines_file)
    if label not in baselines:
        logger.info("remove_baseline: label '%s' not found.", label)
        return False
    del baselines[label]
    _save_baselines(baselines, baselines_file)
    logger.info("remove_baseline: removed label '%s'.", label)
    return True
