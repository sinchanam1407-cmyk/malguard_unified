"""
signature_check.py — APK digital signature verification.

Attempts to verify an APK's digital signature using:
  1. ``apksigner`` (preferred, part of Android SDK Build-Tools)
  2. ``jarsigner`` (JDK, fallback)

If neither tool is available the function returns a status of
``"tool_unavailable"`` so the rest of the analysis can continue.

.. important::
   This module **never executes** the APK being analysed.  Only the
   verification utilities are invoked, and they are passed the APK path as a
   read-only argument.

Usage::

    from signature_check import verify_apk_signature, is_valid_apk

    result = verify_apk_signature("app.apk")
    print(result["status"])   # "verified" | "failed" | "unsigned" | ...
"""

import logging
import os
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)

# Maximum characters of tool output to include in the result
_MAX_OUTPUT_SNIPPET = 2000


def is_valid_apk(file_path: str) -> bool:
    """
    Perform a lightweight sanity-check on *file_path* to confirm it looks like
    a valid APK before attempting signature verification.

    Checks:
    * The filename ends with ``.apk`` (case-insensitive).
    * The file begins with the ZIP magic bytes ``PK\\x03\\x04``.

    Parameters
    ----------
    file_path : str
        Path to the file to inspect.

    Returns
    -------
    bool
        ``True`` if both checks pass; ``False`` otherwise.
    """
    if not str(file_path).lower().endswith(".apk"):
        return False
    try:
        with open(file_path, "rb") as fh:
            magic = fh.read(4)
        return magic == b"PK\x03\x04"
    except OSError:
        return False


def _run_tool(cmd: list) -> tuple[int, str, str]:
    """
    Run *cmd* and return (returncode, stdout, stderr).

    All subprocess exceptions are caught and re-raised so callers can handle
    them consistently.
    """
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.returncode, result.stdout, result.stderr


def verify_apk_signature(apk_path: str) -> dict:
    """
    Verify the digital signature of an APK file using available system tools.

    The function tries ``apksigner`` first.  If that is not found it falls back
    to ``jarsigner``.  If neither is available it returns
    ``status="tool_unavailable"`` so the caller can handle that gracefully.

    .. note::
       The APK file is **never executed**.  Only the verification tool is run
       against it in read-only mode.

    Parameters
    ----------
    apk_path : str
        Path to the ``.apk`` file to verify.

    Returns
    -------
    dict
        A result dictionary with keys:

        ``status``
            ``"verified"`` – signature is valid and trusted.
            ``"failed"``   – tool ran but signature could not be verified.
            ``"unsigned"`` – APK has no signature block.
            ``"tool_unavailable"`` – neither ``apksigner`` nor ``jarsigner``
            is on the system PATH.
            ``"error"``    – an unexpected exception occurred.
        ``tool_used``
            Name of the tool that was used (or ``"none"``).
        ``certificate_info``
            Raw output snippet from the verification tool.
        ``message``
            Human-readable description of the outcome.
    """
    # ── Try apksigner ─────────────────────────────────────────────────────────
    try:
        rc, stdout, stderr = _run_tool(
            ["apksigner", "verify", "--verbose", "--print-certs", apk_path]
        )
        combined = (stdout + "\n" + stderr).strip()
        snippet = combined[:_MAX_OUTPUT_SNIPPET]

        if rc == 0 and "Verified using" in combined:
            return {
                "status": "verified",
                "tool_used": "apksigner",
                "certificate_info": snippet,
                "message": "APK signature verified successfully by apksigner.",
            }
        if "does not verify" in combined.lower() or "unsigned" in combined.lower():
            status = "unsigned" if "unsigned" in combined.lower() else "failed"
            return {
                "status": status,
                "tool_used": "apksigner",
                "certificate_info": snippet,
                "message": f"apksigner: {status}.",
            }
        # Non-zero exit but not specifically unsigned
        return {
            "status": "failed",
            "tool_used": "apksigner",
            "certificate_info": snippet,
            "message": f"apksigner exited with code {rc}.",
        }

    except FileNotFoundError:
        logger.debug("verify_apk_signature: apksigner not found, trying jarsigner.")
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "tool_used": "apksigner",
            "certificate_info": "",
            "message": "apksigner timed out after 30 seconds.",
        }
    except Exception as exc:
        logger.warning("verify_apk_signature: apksigner error: %s", exc)

    # ── Try jarsigner ─────────────────────────────────────────────────────────
    try:
        rc, stdout, stderr = _run_tool(["jarsigner", "-verify", apk_path])
        combined = (stdout + "\n" + stderr).strip()
        snippet = combined[:_MAX_OUTPUT_SNIPPET]

        if rc == 0 and "verified" in combined.lower():
            return {
                "status": "verified",
                "tool_used": "jarsigner",
                "certificate_info": snippet,
                "message": "APK signature verified successfully by jarsigner.",
            }
        return {
            "status": "failed",
            "tool_used": "jarsigner",
            "certificate_info": snippet,
            "message": f"jarsigner verification failed (exit code {rc}).",
        }

    except FileNotFoundError:
        logger.debug("verify_apk_signature: jarsigner not found.")
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "tool_used": "jarsigner",
            "certificate_info": "",
            "message": "jarsigner timed out after 30 seconds.",
        }
    except Exception as exc:
        logger.warning("verify_apk_signature: jarsigner error: %s", exc)

    # ── Neither tool is available ─────────────────────────────────────────────
    return {
        "status": "tool_unavailable",
        "tool_used": "none",
        "certificate_info": "",
        "message": (
            "No APK signature verification tool found. "
            "Install Android SDK Build-Tools (apksigner) or JDK (jarsigner)."
        ),
    }
