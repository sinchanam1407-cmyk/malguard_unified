"""
feature_extractor.py — Static feature extraction from Android APK files.

Uses APKTool + Baksmali (via subprocess) to decompile APKs, then parses:
  - AndroidManifest.xml  → permissions, activities, services, receivers, intents
  - Smali bytecode       → API call patterns

Output: a single feature dict per APK (binary presence/absence vectors).

Usage:
    extractor = FeatureExtractor(apk_path="app.apk")
    features  = extractor.extract()
"""

import os
import re
import subprocess
import shutil
import tempfile
import xml.etree.ElementTree as ET
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# ── Known dangerous permissions (common in malware) ───────────────────────────
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

# ── Known malicious API call patterns ─────────────────────────────────────────
KNOWN_APIS = [
    "Landroid/telephony/SmsManager;->sendTextMessage",
    "Landroid/telephony/TelephonyManager;->getDeviceId",
    "Landroid/telephony/TelephonyManager;->getSubscriberId",
    "Landroid/content/ContentResolver;->query",
    "Ljava/lang/Runtime;->exec",
    "Ljava/lang/reflect/Method;->invoke",
    "Ldalvik/system/DexClassLoader;->",
    "Ldalvik/system/PathClassLoader;->",
    "Landroid/app/ActivityManager;->getRunningAppProcesses",
    "Ljava/net/HttpURLConnection;->",
    "Ljavax/net/ssl/",
    "Landroid/location/LocationManager;->getLastKnownLocation",
    "Landroid/content/pm/PackageManager;->getInstalledPackages",
    "Landroid/net/wifi/WifiManager;->",
    "Landroid/os/PowerManager;->newWakeLock",
    "Ljava/io/FileOutputStream;->",
    "Ljava/io/FileInputStream;->",
    "Landroid/accounts/AccountManager;->",
    "Landroid/content/Context;->sendBroadcast",
    "Landroid/app/NotificationManager;->",
]

# ── Known suspicious intents ──────────────────────────────────────────────────
KNOWN_INTENTS = [
    "android.intent.action.BOOT_COMPLETED",
    "android.intent.action.SMS_RECEIVED",
    "android.intent.action.PACKAGE_ADDED",
    "android.intent.action.PACKAGE_REMOVED",
    "android.intent.action.SCREEN_ON",
    "android.intent.action.SCREEN_OFF",
    "android.intent.action.PHONE_STATE",
    "android.intent.action.NEW_OUTGOING_CALL",
]


class FeatureExtractor:
    """
    Extracts static features from an Android APK file.

    Parameters
    ----------
    apk_path : str
        Path to the .apk file.
    apktool_path : str
        Path to the apktool jar or wrapper script.
    baksmali_path : str
        Path to the baksmali jar (optional; APKTool produces smali already).
    """

    def __init__(
        self,
        apk_path: str,
        apktool_path: str = "apktool",
        baksmali_path: str = "baksmali",
    ):
        self.apk_path      = apk_path
        self.apktool_path  = apktool_path
        self.baksmali_path = baksmali_path
        self._tmpdir: str  = ""

    # ── public ────────────────────────────────────────────────────────────────
    def extract(self) -> Dict[str, Any]:
        """
        Decompile the APK, extract all features, and return a feature dict.
        Keys are feature names; values are 0/1 (presence) or integer counts.
        """
        self._tmpdir = tempfile.mkdtemp(prefix="apk_extract_")
        try:
            self._decompile()
            features = {}
            features.update(self._extract_manifest_features())
            features.update(self._extract_api_features())
            return features
        finally:
            shutil.rmtree(self._tmpdir, ignore_errors=True)

    # ── private helpers ───────────────────────────────────────────────────────
    def _decompile(self):
        """Run APKTool to decompile the APK into self._tmpdir."""
        out_dir = os.path.join(self._tmpdir, "decompiled")
        cmd = [
            self.apktool_path, "d",
            self.apk_path,
            "-o", out_dir,
            "--force",
            "--no-res",        # skip resource decoding for speed
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            logger.warning(
                "APKTool failed for %s: %s", self.apk_path, result.stderr[:200]
            )
        self._decompiled_dir = out_dir

    def _extract_manifest_features(self) -> Dict[str, int]:
        """Parse AndroidManifest.xml for permissions, components, and intents."""
        features: Dict[str, int] = {}

        # Zero-initialise all known features
        for p in KNOWN_PERMISSIONS:
            features[f"perm_{p.split('.')[-1]}"] = 0
        for intent in KNOWN_INTENTS:
            features[f"intent_{intent.split('.')[-1]}"] = 0
        features["num_activities"]  = 0
        features["num_services"]    = 0
        features["num_receivers"]   = 0
        features["num_providers"]   = 0
        features["num_permissions"] = 0

        manifest_path = os.path.join(self._decompiled_dir, "AndroidManifest.xml")
        if not os.path.exists(manifest_path):
            logger.warning("Manifest not found at %s", manifest_path)
            return features

        try:
            tree = ET.parse(manifest_path)
            root = tree.getroot()
            ns   = {"android": "http://schemas.android.com/apk/res/android"}

            # Permissions
            for el in root.findall("uses-permission"):
                pname = el.get("{http://schemas.android.com/apk/res/android}name", "")
                key = f"perm_{pname.split('.')[-1]}"
                if key in features:
                    features[key] = 1
                features["num_permissions"] += 1

            # Components
            app = root.find("application")
            if app is not None:
                features["num_activities"] = len(app.findall("activity"))
                features["num_services"]   = len(app.findall("service"))
                features["num_receivers"]  = len(app.findall("receiver"))
                features["num_providers"]  = len(app.findall("provider"))

                # Intent filters
                for el in app.iter("action"):
                    iname = el.get(
                        "{http://schemas.android.com/apk/res/android}name", ""
                    )
                    key = f"intent_{iname.split('.')[-1]}"
                    if key in features:
                        features[key] = 1

        except ET.ParseError as exc:
            logger.warning("Manifest parse error for %s: %s", self.apk_path, exc)

        return features

    def _extract_api_features(self) -> Dict[str, int]:
        """Scan smali files for known API call patterns."""
        features: Dict[str, int] = {
            f"api_{i}": 0 for i in range(len(KNOWN_APIS))
        }
        features["total_smali_files"] = 0
        features["total_invoke_calls"] = 0

        smali_root = os.path.join(self._decompiled_dir, "smali")
        if not os.path.exists(smali_root):
            return features

        for dirpath, _, filenames in os.walk(smali_root):
            for fname in filenames:
                if not fname.endswith(".smali"):
                    continue
                features["total_smali_files"] += 1
                fpath = os.path.join(dirpath, fname)
                try:
                    content = open(fpath, "r", errors="ignore").read()
                    features["total_invoke_calls"] += content.count("invoke-")
                    for i, api in enumerate(KNOWN_APIS):
                        if api in content:
                            features[f"api_{i}"] = 1
                except OSError:
                    pass

        return features


# ── Dataset-level extraction ───────────────────────────────────────────────────
def extract_dataset(apk_dir: str, label_file: str, output_csv: str):
    """
    Iterate over APKs listed in label_file, extract features for each,
    and write the feature matrix to output_csv.

    Parameters
    ----------
    apk_dir    : directory containing .apk files
    label_file : CSV with columns [apk_name, label]
    output_csv : path to write the resulting feature matrix
    """
    import pandas as pd

    labels_df = pd.read_csv(label_file)
    rows = []

    for _, row in labels_df.iterrows():
        apk_path = os.path.join(apk_dir, row["apk_name"])
        if not os.path.exists(apk_path):
            logger.warning("APK not found: %s — skipping", apk_path)
            continue

        logger.info("Extracting: %s", row["apk_name"])
        try:
            feats = FeatureExtractor(apk_path).extract()
            feats["apk_name"] = row["apk_name"]
            feats["label"]    = int(row["label"])
            rows.append(feats)
        except Exception as exc:
            logger.error("Error extracting %s: %s", row["apk_name"], exc)

    df = pd.DataFrame(rows).fillna(0)
    df.to_csv(output_csv, index=False)
    logger.info("Saved feature matrix to %s (%d samples)", output_csv, len(df))
    return df
