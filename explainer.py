"""
explainer.py — SHAP-based explainability for the best ML model.

Provides:
  - Global feature importance (SHAP bar + beeswarm plots)
  - Local explanation for a single APK prediction
  - Text report of the top contributing features

Usage:
    explainer = SHAPExplainer(model, X_train_sample, feature_names, report_dir)
    explainer.explain_global(X_test)
    report = explainer.explain_local(X_test.iloc[0])
    print(report)
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

DARK_BG  = "#05060c"
TEXT_CLR = "#eef4ff"
SUB_CLR  = "#64748b"


def _set_style():
    plt.rcParams.update({
        "figure.facecolor":  DARK_BG,
        "axes.facecolor":    "#0e1622",
        "axes.edgecolor":    "#1a2235",
        "axes.labelcolor":   TEXT_CLR,
        "text.color":        TEXT_CLR,
        "xtick.color":       SUB_CLR,
        "ytick.color":       SUB_CLR,
        "grid.color":        "#1a2235",
        "font.family":       "DejaVu Sans",
        "font.size":         10,
    })


class SHAPExplainer:
    """
    Wraps SHAP for tree-based and other scikit-learn models.

    Parameters
    ----------
    model : fitted sklearn estimator
    X_background : array-like, shape (n_bg, n_features)
        Small background dataset for SHAP baseline (training sample).
    feature_names : list of str
    report_dir : str
        Directory to save SHAP plots and text report.
    max_bg_samples : int
        Maximum background samples to use (keep small for speed).
    top_n : int
        Number of top features to show in global plot.
    """

    def __init__(
        self,
        model,
        X_background: np.ndarray,
        feature_names: list,
        report_dir: str,
        max_bg_samples: int = 100,
        top_n: int = 15,
    ):
        try:
            import shap
            self._shap = shap
        except ImportError:
            raise ImportError(
                "SHAP is not installed. Run: pip install shap"
            )

        self.model         = model
        self.feature_names = feature_names
        self.report_dir    = report_dir
        self.top_n         = top_n
        os.makedirs(report_dir, exist_ok=True)
        _set_style()

        # Background dataset
        n_bg = min(max_bg_samples, len(X_background))
        rng  = np.random.default_rng(42)
        idx  = rng.choice(len(X_background), n_bg, replace=False)
        self._bg = X_background[idx]

        # Choose SHAP explainer type
        model_type = type(model).__name__
        if model_type in (
            "RandomForestClassifier",
            "GradientBoostingClassifier",
            "AdaBoostClassifier",
        ):
            self._explainer = shap.TreeExplainer(model)
            logger.info("Using SHAP TreeExplainer for %s", model_type)
        else:
            self._explainer = shap.KernelExplainer(
                model.predict_proba, self._bg
            )
            logger.info("Using SHAP KernelExplainer for %s", model_type)

    # ── global explanation ────────────────────────────────────────────────────
    def explain_global(self, X_test: np.ndarray):
        """
        Compute SHAP values for X_test and save global feature importance plots.
        """
        logger.info("Computing SHAP values for %d samples …", len(X_test))
        shap_values = self._explainer.shap_values(X_test)

        # For binary classifiers, shap_values may be a list [class0, class1]
        if isinstance(shap_values, list):
            sv = shap_values[1]   # malware class
        else:
            sv = shap_values

        # Mean |SHAP| per feature
        mean_abs = np.abs(sv).mean(axis=0)
        top_idx  = np.argsort(mean_abs)[::-1][: self.top_n]
        top_feats = [self.feature_names[i] for i in top_idx]
        top_vals  = mean_abs[top_idx]

        # Bar plot
        fig, ax = plt.subplots(figsize=(10, 0.45 * self.top_n + 2))
        colors = [
            "#f87171" if v > np.median(top_vals) else "#a78bfa"
            for v in top_vals[::-1]
        ]
        ax.barh(range(self.top_n), top_vals[::-1], color=colors, edgecolor="none", height=0.65)
        ax.set_yticks(range(self.top_n))
        ax.set_yticklabels(top_feats[::-1], fontsize=9)
        ax.set_xlabel("Mean |SHAP value| — Impact on Malware Prediction", color=TEXT_CLR)
        ax.set_title(
            f"SHAP Global Feature Importance — Top {self.top_n} Features",
            color=TEXT_CLR, fontsize=12,
        )
        ax.grid(True, axis="x", alpha=0.3)
        ax.set_axisbelow(True)

        for i, val in enumerate(top_vals[::-1]):
            ax.text(val + 0.0005, i, f"{val:.4f}", va="center", fontsize=8, color=TEXT_CLR)

        path = os.path.join(self.report_dir, "shap_global.png")
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        plt.close()
        logger.info("Saved SHAP global plot → %s", path)

        self._global_shap_values  = sv
        self._global_top_features = list(zip(top_feats, top_vals))
        return sv

    # ── local explanation ─────────────────────────────────────────────────────
    def explain_local(self, x_single: np.ndarray) -> str:
        """
        Explain a single APK prediction.

        Parameters
        ----------
        x_single : 1D array of feature values for one APK.

        Returns
        -------
        str : Human-readable explanation report.
        """
        sv = self._explainer.shap_values(x_single.reshape(1, -1))
        if isinstance(sv, list):
            sv = sv[1]
        sv = sv.flatten()

        # Rank by absolute SHAP value
        idx = np.argsort(np.abs(sv))[::-1][: self.top_n]
        pred_label = self.model.predict(x_single.reshape(1, -1))[0]
        pred_prob  = self.model.predict_proba(x_single.reshape(1, -1))[0][1]

        lines = [
            "=" * 60,
            f"APK PREDICTION REPORT",
            f"Prediction : {'MALWARE ⚠' if pred_label == 1 else 'BENIGN ✓'}",
            f"Confidence : {pred_prob*100:.1f}%",
            "=" * 60,
            f"Top contributing features:",
            "-" * 60,
        ]
        for rank, i in enumerate(idx, 1):
            direction = "▲ MALWARE" if sv[i] > 0 else "▼ BENIGN"
            lines.append(
                f"  {rank:>2}. {self.feature_names[i]:<40}"
                f"  SHAP={sv[i]:+.4f}  {direction}"
            )
        lines.append("=" * 60)
        report = "\n".join(lines)
        print(report)

        # Save text report
        path = os.path.join(self.report_dir, "shap_local_explanation.txt")
        with open(path, "w") as f:
            f.write(report)
        logger.info("Saved local SHAP explanation → %s", path)
        return report
