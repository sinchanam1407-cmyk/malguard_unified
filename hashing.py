"""
hashing.py — SHA-256 cryptographic hashing utilities for file integrity.

Provides functions for hashing files and raw bytes using SHA-256, and for
verifying file integrity against a known-good hash value.

Usage:
    from hashing import hash_file, hash_bytes, verify_hash

    digest = hash_file("app.apk")
    ok = verify_hash("app.apk", digest)
"""

import hashlib
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Chunk size used when reading files (8 KiB)
_CHUNK_SIZE: int = 8192


def hash_file(file_path: str) -> str:
    """
    Compute the SHA-256 digest of a file by reading it in chunks.

    Parameters
    ----------
    file_path : str
        Absolute or relative path to the file to hash.

    Returns
    -------
    str
        Lowercase hexadecimal SHA-256 digest (64 characters).

    Raises
    ------
    FileNotFoundError
        If *file_path* does not exist.
    OSError
        If the file cannot be read.
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as fh:
        while chunk := fh.read(_CHUNK_SIZE):
            sha256.update(chunk)
    return sha256.hexdigest()


def hash_bytes(data: bytes) -> str:
    """
    Compute the SHA-256 digest of a raw byte sequence.

    Parameters
    ----------
    data : bytes
        The bytes to hash.

    Returns
    -------
    str
        Lowercase hexadecimal SHA-256 digest (64 characters).
    """
    return hashlib.sha256(data).hexdigest()


def verify_hash(file_path: str, expected_hash: str) -> bool:
    """
    Check whether a file's SHA-256 hash matches an expected value.

    Comparison is case-insensitive so both upper- and lower-case hex strings
    are accepted.

    Parameters
    ----------
    file_path : str
        Path to the file to verify.
    expected_hash : str
        The expected SHA-256 hex digest to compare against.

    Returns
    -------
    bool
        ``True`` if the computed digest matches *expected_hash*; ``False``
        otherwise (including on I/O errors, which are logged as warnings).
    """
    try:
        actual = hash_file(file_path)
        return actual.lower() == expected_hash.lower()
    except (FileNotFoundError, OSError) as exc:
        logger.warning("verify_hash: could not read '%s': %s", file_path, exc)
        return False
