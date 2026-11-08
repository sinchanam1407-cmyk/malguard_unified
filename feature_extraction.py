"""
feature_extraction.py — Simplified static feature extraction for a single APK.

This module is a standalone wrapper that extracts the 58 binary features
(30 permissions + 20 API call patterns + 8 intents) used by the trained
GradientBoosting model.

Extraction strategy (in order of preference):
  1. Use ``androguard`` if installed.
  2. Fall back to reading the raw ZIP contents (AndroidManifest.xml binary
     blob + smali files if present).

On any unrecoverable error the function returns a dict of all-zero features
plus an ``extraction_error`` key describing what went wrong.

Usage::

    from feature_extraction import extract_features_from_apk, get_feature_vector
    from malware_prediction import FEATURE_NAMES

    features = extract_features_from_apk("app.apk")
    vector   = get_feature_vector(features, FEATURE_NAMES)
"""

import logging
import os
import re
import zipfile
from typing import Any

logger = logging.getLogger(__name__)

# ── Feature lists ──────────────────────────────────────────────────────────────

KNOWN_PERMISSIONS = [
    "android.permission.READ_SMS",
    "android.permission.SEND_SMS",
    "android.permission.RECEIVE_SMS",
    "android.permission.READ_CONTACTS",
    "android.permission.WRITE_CONTACTS",
    "android.permission.READ_CALL_LOG",
    "android.permission.CALL_PHONE",
    "android.permission.INTERNET",
    "android.permission.ACCESS_FINE_LOCATION",
    "android.permission.ACCESS_COARSE_LOCATION",
    "android.permission.CAMERA",
    "android.permission.RECORD_AUDIO",
    "android.permission.READ_EXTERNAL_STORAGE",
    "android.permission.WRITE_EXTERNAL_STORAGE",
    "android.permission.RECEIVE_BOOT_COMPLETED",
    "android.permission.SYSTEM_ALERT_WINDOW",
    "android.permission.GET_ACCOUNTS",
    "android.permission.USE_CREDENTIALS",
    "android.permission.AUTHENTICATE_ACCOUNTS",
    "android.permission.CHANGE_NETWORK_STATE",
    "android.permission.ACCESS_WIFI_STATE",
    "android.permission.CHANGE_WIFI_STATE",
    "android.permission.BLUETOOTH",
    "android.permission.VIBRATE",
    "android.permission.WAKE_LOCK",
    "android.permission.INSTALL_PACKAGES",
    "android.permission.DELETE_PACKAGES",
    "android.permission.KILL_BACKGROUND_PROCESSES",
    "android.permission.PROCESS_OUTGOING_CALLS",
    "android.permission.READ_PHONE_STATE",
]

KNOWN_API_CALLS = [
    "Ljava/lang/Runtime;->exec",
    "Ljava/lang/ProcessBuilder",
    "Ljava/net/HttpURLConnection",
    "Landroid/telephony/SmsManager;->sendTextMessage",
    "Ldalvik/system/DexClassLoader",
    "Ljava/lang/reflect/Method;->invoke",
    "Landroid/content/pm/PackageManager",
    "Ljava/net/URL;->openConnection",
    "Landroid/os/Environment;->getExternalStorageDirectory",
    "Landroid/content/ContentResolver",
    "Landroid/location/LocationManager",
    "Landroid/telephony/TelephonyManager;->getDeviceId",
    "Landroid/media/AudioRecord",
    "Landroid/hardware/Camera",
    "Ljava/net/Socket",
    "Lorg/apache/http/impl/client/DefaultHttpClient",
    "Landroid/os/Binder",
    "Ljava/util/zip/ZipInputStream",
    "Landroid/app/admin/DevicePolicyManager",
    "Ljavax/crypto/Cipher",
]

KNOWN_INTENTS = [
    "android.intent.action.BOOT_COMPLETED",
    "android.intent.action.SMS_RECEIVED",
    "android.intent.action.PACKAGE_REPLACED",
    "android.intent.action.PACKAGE_ADDED",
    "android.telephony.action.CARRIER_CONFIG_CHANGED",
    "android.intent.action.CALL",
    "android.intent.action.SEND",
    "android.intent.action.VIEW",
]


def _zero_features() -> dict:
    """Return a dict of all 58 features initialised to 0."""
    features: dict = {}
    for p in KNOWN_PERMISSIONS:
        features[f"perm_{p.split('.')[-1]}"] = 0
    for api in KNOWN_API_CALLS:
        features[f"api_{api.split(';')[0].split('/')[-1].split('->')[0]}"] = 0
    for intent in KNOWN_INTENTS:
        features[f"intent_{intent.split('.')[-1]}"] = 0
    return features


# Stable feature name builders used by FEATURE_NAMES in malware_prediction.py
def _perm_key(p: str) -> str:
    return f"perm_{p.split('.')[-1]}"


def _api_key(api: str) -> str:
    base = api.split(";")[0].split("/")[-1]
    if "->" in base:
        base = base.split("->")[0]
    return f"api_{base}"


def _intent_key(intent: str) -> str:
    return f"intent_{intent.split('.')[-1]}"


# ── Androguard-based extraction ────────────────────────────────────────────────

def _extract_with_androguard(apk_path: str) -> dict:
    """Use androguard to extract features from the APK."""
    from androguard.misc import AnalyzeAPK  # type: ignore

    a, d, dx = AnalyzeAPK(apk_path)
    features = _zero_features()

    # Permissions
    declared = set(a.get_permissions())
    for p in KNOWN_PERMISSIONS:
        if p in declared:
            features[_perm_key(p)] = 1

    # Smali / API patterns — scan all disassembled code
    all_smali_text = ""
    for dex in (d if isinstance(d, list) else [d]):
        try:
            for cls in dex.get_classes():
                for method in cls.get_methods():
                    code = method.get_code()
                    if code:
                        all_smali_text += str(code.get_bc()) + "\n"
        except Exception:
            pass

    for api in KNOWN_API_CALLS:
        if api in all_smali_text:
            features[_api_key(api)] = 1

    # Intents — from manifest receivers / services / activities
    manifest_text = str(a.get_android_manifest_axml().get_xml())
    for intent in KNOWN_INTENTS:
        if intent in manifest_text:
            features[_intent_key(intent)] = 1

    return features


# ── Zip/raw-bytes fallback extraction ─────────────────────────────────────────

def _bytes_contain(haystack: bytes, needle: str) -> bool:
    return needle.encode("utf-8") in haystack or needle.encode("utf-16-le") in haystack


def _extract_with_zipfile(apk_path: str) -> dict:
    """
    Fall-back extraction using Python's zipfile module.

    Reads the raw bytes of AndroidManifest.xml (binary AXML) and any
    *.smali files embedded in the archive.  The binary manifest is searched
    for permission/intent string patterns using both UTF-8 and UTF-16-LE
    encodings because AXML stores strings in a string pool.
    """
    features = _zero_features()

    with zipfile.ZipFile(apk_path, "r") as zf:
        names = set(zf.namelist())

        # ── AndroidManifest.xml ────────────────────────────────────────────
        if "AndroidManifest.xml" in names:
            manifest_bytes = zf.read("AndroidManifest.xml")
            for p in KNOWN_PERMISSIONS:
                if _bytes_contain(manifest_bytes, p):
                    features[_perm_key(p)] = 1
            for intent in KNOWN_INTENTS:
                if _bytes_contain(manifest_bytes, intent):
                    features[_intent_key(intent)] = 1

        # ── Smali files (if present — e.g. pre-processed APK) ─────────────
        smali_files = [n for n in names if n.endswith(".smali")]
        if smali_files:
            for smali_name in smali_files:
                try:
                    content = zf.read(smali_name).decode("utf-8", errors="ignore")
                    for api in KNOWN_API_CALLS:
                        if api in content:
                            features[_api_key(api)] = 1
                except Exception:
                    pass
        else:
            # Try classes.dex — scan raw bytes for API string patterns
            dex_files = [n for n in names if re.match(r"classes\d*\.dex$", n)]
            for dex_name in dex_files:
                try:
                    dex_bytes = zf.read(dex_name)
                    for api in KNOWN_API_CALLS:
                        if _bytes_contain(dex_bytes, api):
                            features[_api_key(api)] = 1
                except Exception:
                    pass

    return features


# ── Public API ─────────────────────────────────────────────────────────────────

def extract_features_from_apk(apk_path: str) -> dict:
    """
    Extract binary (0/1) features from an APK file.

    Tries androguard first; falls back to the zipfile approach.  On any
    unrecoverable error returns all-zero features with an ``extraction_error``
    key.

    Parameters
    ----------
    apk_path : str
        Path to the ``.apk`` file.

    Returns
    -------
    dict
        Mapping of feature name → 0 or 1 for all 58 features.
        May include an extra ``extraction_error`` key if both extraction
        methods failed.
    """
    # ── Try androguard ────────────────────────────────────────────────────
    try:
        import androguard  # noqa: F401
        logger.info("extract_features_from_apk: using androguard for '%s'", apk_path)
        return _extract_with_androguard(apk_path)
    except ImportError:
        logger.info("extract_features_from_apk: androguard not available, using zipfile fallback.")
    except Exception as exc:
        logger.warning("extract_features_from_apk: androguard failed (%s), trying zipfile.", exc)

    # ── Try zipfile fallback ──────────────────────────────────────────────
    try:
        logger.info("extract_features_from_apk: using zipfile fallback for '%s'", apk_path)
        return _extract_with_zipfile(apk_path)
    except Exception as exc:
        logger.error("extract_features_from_apk: zipfile extraction failed: %s", exc)
        zero = _zero_features()
        zero["extraction_error"] = str(exc)
        return zero


def get_feature_vector(features_dict: dict, feature_names: list) -> list:
    """
    Convert a features dict to an ordered numeric list.

    Parameters
    ----------
    features_dict : dict
        Mapping of feature name → value (typically 0 or 1).
    feature_names : list
        Ordered list of feature names defining the vector layout.

    Returns
    -------
    list
        Values in the same order as *feature_names*. Any missing feature
        defaults to ``0``.
    """
    return [features_dict.get(name, 0) for name in feature_names]
