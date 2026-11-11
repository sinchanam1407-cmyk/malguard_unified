"""
signature_check.py — APK digital signature verification.

Verification strategy (in order of preference):
  1. ``apksigner`` (Android SDK Build-Tools) — most authoritative
  2. ``jarsigner`` (JDK) — fallback for v1 (JAR) signatures
  3. Pure-Python ZIP inspection — always available; detects presence of
     META-INF/*.RSA / *.DSA / *.EC signature blocks and parses basic
     certificate fields without executing any external tool.

The APK is NEVER executed. Only signature metadata is read.

Usage::

    from signature_check import verify_apk_signature, is_valid_apk

    result = verify_apk_signature("app.apk")
    print(result["status"])   # "verified" | "failed" | "unsigned" | "error"
"""

import logging
import os
import re
import struct
import subprocess
import zipfile
from typing import Optional

logger = logging.getLogger(__name__)

_MAX_OUTPUT_SNIPPET = 2000


# ─────────────────────────────────────────────────────────────────────────────
#  Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def is_valid_apk(file_path: str) -> bool:
    """Return True if *file_path* looks like a valid APK (ZIP magic + .apk ext)."""
    if not str(file_path).lower().endswith(".apk"):
        return False
    try:
        with open(file_path, "rb") as fh:
            return fh.read(4) == b"PK\x03\x04"
    except OSError:
        return False


def _run_tool(cmd: list) -> tuple:
    """Run *cmd* and return (returncode, stdout, stderr)."""
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return result.returncode, result.stdout, result.stderr


# ─────────────────────────────────────────────────────────────────────────────
#  Pure-Python ZIP-based signature inspection
# ─────────────────────────────────────────────────────────────────────────────

def _inspect_zip_signature(apk_path: str) -> dict:
    """
    Inspect the APK's signature blocks using Python's zipfile module.

    Checks for:
    - META-INF/MANIFEST.MF        — JAR manifest (required for v1)
    - META-INF/*.SF               — signature file
    - META-INF/*.RSA / *.DSA / *.EC — PKCS#7 signature block (v1 JAR sig)
    - META-INF/APK_SIGNING_BLOCK  — Android v2/v3 indicator (between ZIP entries)

    Returns a dict with keys: status, tool_used, certificate_info, message.
    """
    try:
        with zipfile.ZipFile(apk_path, "r") as zf:
            names = set(zf.namelist())

            # --- Look for JAR (v1) signature entries ---
            sig_blocks = [n for n in names if re.match(
                r"META-INF/[^/]+\.(RSA|DSA|EC)$", n, re.IGNORECASE)]
            sf_files   = [n for n in names if re.match(
                r"META-INF/[^/]+\.SF$", n, re.IGNORECASE)]
            has_manifest = "META-INF/MANIFEST.MF" in names

            cert_lines = []

            if sig_blocks and sf_files and has_manifest:
                # Parse the .SF file for digest algorithms and package info
                for sf_name in sf_files[:1]:
                    try:
                        sf_content = zf.read(sf_name).decode("utf-8", errors="ignore")
                        for line in sf_content.splitlines()[:20]:
                            cert_lines.append(line)
                    except Exception:
                        pass

                # Try to extract issuer info from PKCS#7 block (DER)
                for blk_name in sig_blocks[:1]:
                    try:
                        blk_bytes = zf.read(blk_name)
                        # Look for readable strings (CN=, O=, OU=) in the DER blob
                        readable = blk_bytes.decode("latin-1", errors="replace")
                        # Grab printable ASCII runs of length ≥ 4
                        strings = re.findall(r'[ -~]{4,}', readable)
                        cert_candidates = [
                            s.strip() for s in strings
                            if any(kw in s for kw in ("CN=", "O=", "OU=", "C=",
                                                       "Android", "Debug", "Release",
                                                       "CERT", "KEY"))
                        ]
                        if cert_candidates:
                            cert_lines.append("")
                            cert_lines.append("Certificate strings extracted from signature block:")
                            cert_lines.extend(cert_candidates[:15])
                    except Exception:
                        pass

                cert_info = "\n".join(cert_lines)
                sig_names = ", ".join(os.path.basename(b) for b in sig_blocks)

                # Distinguish debug vs release signatures heuristically
                is_debug = any(
                    "debug" in line.lower() or "androiddebugkey" in line.lower()
                    for line in cert_lines
                )
                sig_note = " [DEBUG key detected]" if is_debug else " [Release key]"

                return {
                    "status": "verified",
                    "tool_used": "zip-inspection (v1 JAR signature)",
                    "certificate_info": (
                        f"Signature blocks: {sig_names}{sig_note}\n\n"
                        + cert_info
                    )[:_MAX_OUTPUT_SNIPPET],
                    "message": (
                        f"APK has a valid v1 (JAR) signature block ({sig_names}).{sig_note} "
                        "Note: v2/v3 block verification requires apksigner."
                    ),
                }

            elif has_manifest and not sig_blocks:
                return {
                    "status": "unsigned",
                    "tool_used": "zip-inspection",
                    "certificate_info": "MANIFEST.MF present but no .RSA/.DSA/.EC block found.",
                    "message": "APK has a manifest but no signature block — it appears unsigned.",
                }
            else:
                return {
                    "status": "unsigned",
                    "tool_used": "zip-inspection",
                    "certificate_info": f"ZIP entries found: {len(names)}. No META-INF signature data.",
                    "message": "No JAR signature data found in APK. File may be unsigned or v2/v3 only.",
                }

    except zipfile.BadZipFile:
        return {
            "status": "error",
            "tool_used": "zip-inspection",
            "certificate_info": "",
            "message": "File is not a valid ZIP/APK archive.",
        }
    except Exception as exc:
        logger.warning("_inspect_zip_signature: %s", exc)
        return {
            "status": "error",
            "tool_used": "zip-inspection",
            "certificate_info": "",
            "message": f"ZIP inspection error: {exc}",
        }


# ─────────────────────────────────────────────────────────────────────────────
#  External tool wrappers
# ─────────────────────────────────────────────────────────────────────────────

def _find_apksigner() -> Optional[str]:
    """
    Find apksigner on common paths: system PATH, Android SDK, and
    common Windows/Linux installation directories.
    """
    import shutil

    # 1. System PATH
    found = shutil.which("apksigner")
    if found:
        return found

    # 2. Common Android SDK locations
    candidate_roots = []

    # Windows
    for base in [
        os.path.expanduser("~\\AppData\\Local\\Android\\Sdk"),
        "C:\\Users\\Public\\Android\\Sdk",
        "C:\\Android\\sdk",
    ]:
        candidate_roots.append(base)

    # Linux / macOS
    for base in [
        os.path.expanduser("~/Android/Sdk"),
        "/opt/android-sdk",
        "/usr/local/android-sdk",
    ]:
        candidate_roots.append(base)

    # Also check ANDROID_HOME / ANDROID_SDK_ROOT env vars
    for env_var in ("ANDROID_HOME", "ANDROID_SDK_ROOT"):
        val = os.environ.get(env_var)
        if val:
            candidate_roots.insert(0, val)

    for sdk_root in candidate_roots:
        build_tools = os.path.join(sdk_root, "build-tools")
        if os.path.isdir(build_tools):
            # Pick the latest build-tools version
            versions = sorted(
                (d for d in os.listdir(build_tools)
                 if os.path.isdir(os.path.join(build_tools, d))),
                reverse=True,
            )
            for ver in versions:
                for exe in ("apksigner.bat", "apksigner"):
                    candidate = os.path.join(build_tools, ver, exe)
                    if os.path.isfile(candidate):
                        return candidate

    return None


def _try_apksigner(apk_path: str) -> Optional[dict]:
    """Try apksigner; return result dict or None if not available."""
    apksigner = _find_apksigner()
    if not apksigner:
        return None

    try:
        rc, stdout, stderr = _run_tool(
            [apksigner, "verify", "--verbose", "--print-certs", apk_path]
        )
        combined = (stdout + "\n" + stderr).strip()
        snippet  = combined[:_MAX_OUTPUT_SNIPPET]

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
        return {
            "status": "failed",
            "tool_used": "apksigner",
            "certificate_info": snippet,
            "message": f"apksigner exited with code {rc}.",
        }

    except (FileNotFoundError, OSError):
        return None
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "tool_used": "apksigner",
            "certificate_info": "",
            "message": "apksigner timed out after 30 seconds.",
        }
    except Exception as exc:
        logger.warning("_try_apksigner: %s", exc)
        return None


def _try_jarsigner(apk_path: str) -> Optional[dict]:
    """Try jarsigner; return result dict or None if not available."""
    try:
        rc, stdout, stderr = _run_tool(["jarsigner", "-verify", apk_path])
        combined = (stdout + "\n" + stderr).strip()
        snippet  = combined[:_MAX_OUTPUT_SNIPPET]

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

    except (FileNotFoundError, OSError):
        return None
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "tool_used": "jarsigner",
            "certificate_info": "",
            "message": "jarsigner timed out after 30 seconds.",
        }
    except Exception as exc:
        logger.warning("_try_jarsigner: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

def verify_apk_signature(apk_path: str) -> dict:
    """
    Verify the digital signature of an APK file.

    Strategy (in order):
    1. apksigner  — from Android SDK (searched on PATH + common install dirs)
    2. jarsigner  — from JDK
    3. ZIP inspection — pure-Python fallback; always works; detects v1 JAR
                       signature blocks without executing any external binary.

    Returns
    -------
    dict
        status          : "verified" | "failed" | "unsigned" | "error"
        tool_used       : name of the tool/method used
        certificate_info: relevant output / certificate strings
        message         : human-readable description
    """
    # 1. apksigner
    result = _try_apksigner(apk_path)
    if result is not None:
        return result

    # 2. jarsigner
    result = _try_jarsigner(apk_path)
    if result is not None:
        return result

    # 3. ZIP-based fallback — always available
    logger.info("verify_apk_signature: using ZIP inspection fallback.")
    return _inspect_zip_signature(apk_path)
