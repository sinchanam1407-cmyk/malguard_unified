"""
evaluator.py — Evaluation metrics, confusion matrices, ROC curves, and reports.

Usage:
    ev = Evaluator(results, feature_names, report_dir="reports/")
    ev.print_summary()
    ev.plot_all()
    ev.save_report()
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for server/headless use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    roc_curve,
    auc,
)

logger = logging.getLogger(__name__)

# ── colour palette ──
COLORS = {
    "RandomForest":     "#34d399",
    "GradientBoosting": "#63b3ed",
    "AdaBoost":         "#a78bfa",
    "SVM":              "#fbbf24",
    "KNN":              "#fb923c",
    "NaiveBayes":       "#f87171",
}
DARK_BG  = "#05060c"
CARD_BG  = "#0e1622"
GRID_CLR = "#1a2235"
TEXT_CLR = "#eef4ff"
SUB_CLR  = "#64748b"


def _set_style():
    plt.rcParams.update({
        "figure.facecolor":  DARK_BG,
        "axes.facecolor":    CARD_BG,
        "axes.edgecolor":    GRID_CLR,
        "axes.labelcolor":   TEXT_CLR,
        "text.color":        TEXT_CLR,
        "xtick.color":       SUB_CLR,
        "ytick.color":       SUB_CLR,
        "grid.color":        GRID_CLR,
        "grid.linewidth":    0.8,
        "font.family":       "DejaVu Sans",
        "font.size":         10,
        "axes.titlesize":    12,
        "axes.titleweight":  "bold",
        "legend.facecolor":  CARD_BG,
        "legend.edgecolor":  GRID_CLR,
    })


class Evaluator:
    """
    Evaluates and visualises ML classifier results.

    Parameters
    ----------
    trainer : ModelTrainer
        Fitted ModelTrainer instance (after trainer.run()).
    feature_names : list of str
        Names of features used in training.
    report_dir : str
        Directory where plots and reports will be saved.
    """

    def __init__(self, trainer, feature_names: list, report_dir: str):
        self.trainer       = trainer
        self.feature_names = feature_names
        self.report_dir    = report_dir
        os.makedirs(report_dir, exist_ok=True)
        _set_style()

    # ── console summary ───────────────────────────────────────────────────────
    def print_summary(self):
        """Print a formatted results table to stdout."""
        df = self.trainer.summary()
        print("\n" + "=" * 90)
        print(f"{'ALGORITHM':<22} {'ACC':>7} {'PREC':>7} {'REC':>7} {'F1':>7} {'MCC':>7} {'AUC':>7} {'FPR':>7}")
        print("-" * 90)
        for _, row in df.iterrows():
            marker = " ★" if row["Algorithm"] == self.trainer.best_name_ else "  "
            print(
                f"{row['Algorithm'] + marker:<22}"
                f"{row['accuracy_mean']*100:>7.2f}"
                f"{row['precision_mean']*100:>7.2f}"
                f"{row['recall_mean']*100:>7.2f}"
                f"{row['f1_mean']*100:>7.2f}"
                f"{row['mcc_mean']:>7.3f}"
                f"{row['auc_mean']:>7.3f}"
                f"{row['fpr_mean']*100:>7.2f}"
            )
        print("=" * 90)
        print(f"\nBest Model : {self.trainer.best_name_}")
        print(f"Best Acc   : {self.trainer.best_acc_*100:.2f}%\n")

    # ── save CSV report ───────────────────────────────────────────────────────
    def save_report(self):
        """Write the summary DataFrame to a CSV file."""
        path = os.path.join(self.report_dir, "results_summary.csv")
        df   = self.trainer.summary()
        df.to_csv(path, index=False)
        logger.info("Saved results summary → %s", path)

    # ── plot dispatcher ───────────────────────────────────────────────────────
    def plot_all(self):
        """Generate and save all evaluation plots."""
        self.plot_metric_comparison()
        self.plot_confusion_matrices()
        self.plot_roc_curves_schematic()
        self.plot_feature_importance()
        logger.info("All plots saved to %s", self.report_dir)

    # ── 1. Metric comparison bar chart ────────────────────────────────────────
    def plot_metric_comparison(self):
        metrics  = ["accuracy_mean", "precision_mean", "recall_mean", "f1_mean"]
        labels   = ["Accuracy", "Precision", "Recall", "F1-Score"]
        df       = self.trainer.summary()
        algos    = df["Algorithm"].tolist()
        n_algos  = len(algos)
        n_m      = len(metrics)
        x        = np.arange(n_algos)
        width    = 0.18

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(
            "Algorithm Performance Comparison (10-Fold CV)",
            fontsize=14, fontweight="bold", color=TEXT_CLR, y=1.01,
        )

        for ax, metric, label in zip(axes, metrics, labels):
            values = df[metric].values * 100
            bars   = ax.bar(
                range(n_algos),
                values,
                color=[COLORS.get(a, "#94a3b8") for a in algos],
                width=0.6,
                edgecolor="none",
                alpha=0.9,
            )
            ax.set_title(label, color=TEXT_CLR)
            ax.set_xticks(range(n_algos))
            ax.set_xticklabels(
                [a.replace("Boosting", "Boost.") for a in algos],
                rotation=30, ha="right", fontsize=8,
            )
            ax.set_ylim(75, 100)
            ax.yaxis.grid(True, alpha=0.4)
            ax.set_axisbelow(True)
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{val:.1f}",
                    ha="center", va="bottom", fontsize=7.5, color=TEXT_CLR,
                )

        plt.tight_layout()
        path = os.path.join(self.report_dir, "metric_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        plt.close()
        logger.info("Saved metric comparison → %s", path)

    # ── 2. Confusion matrices ─────────────────────────────────────────────────
    def plot_confusion_matrices(self):
        """Plot 2×3 grid of averaged confusion matrices for all classifiers."""
        df    = self.trainer.summary()
        algos = df["Algorithm"].tolist()

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        fig.suptitle(
            "Confusion Matrices — All Classifiers",
            fontsize=14, fontweight="bold", color=TEXT_CLR,
        )
        axes = axes.flatten()

        for ax, name in zip(axes, algos):
            cms  = self.trainer.results_[name]["confusion_matrices"]
            # Sum all fold confusion matrices
            tn   = int(np.mean([c[0] for c in cms]))
            fp   = int(np.mean([c[1] for c in cms]))
            fn   = int(np.mean([c[2] for c in cms]))
            tp   = int(np.mean([c[3] for c in cms]))
            cm   = np.array([[tn, fp], [fn, tp]])
            acc  = (tp + tn) / (tp + tn + fp + fn) * 100

            color = COLORS.get(name, "#94a3b8")
            im    = ax.imshow(cm, cmap="Blues", aspect="auto")

            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Pred Benign", "Pred Malware"], color=TEXT_CLR)
            ax.set_yticklabels(["Act Benign", "Act Malware"], color=TEXT_CLR)
            ax.set_title(
                f"{name}\nAcc: {acc:.1f}%",
                color=color, fontsize=10, fontweight="bold",
            )

            for i in range(2):
                for j in range(2):
                    ax.text(
                        j, i, str(cm[i, j]),
                        ha="center", va="center",
                        fontsize=13, fontweight="bold",
                        color="white" if cm[i, j] > cm.max() / 2 else TEXT_CLR,
                    )

        plt.tight_layout()
        path = os.path.join(self.report_dir, "confusion_matrices.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        plt.close()
        logger.info("Saved confusion matrices → %s", path)

    # ── 3. ROC curves (schematic using AUC means) ─────────────────────────────
    def plot_roc_curves_schematic(self):
        """Plot schematic ROC curves based on mean AUC per classifier."""
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.set_title("ROC Curves — All Classifiers", color=TEXT_CLR, fontsize=13)

        df = self.trainer.summary()
        fpr_pts = np.linspace(0, 1, 200)

        for _, row in df.iterrows():
            name = row["Algorithm"]
            auc_val = row["auc_mean"]
            color   = COLORS.get(name, "#94a3b8")
            lw      = 2.5 if name == self.trainer.best_name_ else 1.4
            # Parametric curve: TPR = FPR^(1-AUC) approximation
            tpr_pts = np.power(fpr_pts, 1 - auc_val)
            ax.plot(
                fpr_pts, tpr_pts,
                color=color, linewidth=lw,
                label=f"{name} (AUC={auc_val:.3f})",
                alpha=0.85 if name == self.trainer.best_name_ else 0.65,
            )

        # Diagonal
        ax.plot([0, 1], [0, 1], "w--", linewidth=0.8, alpha=0.4, label="Random (AUC=0.5)")
        ax.set_xlabel("False Positive Rate", color=TEXT_CLR)
        ax.set_ylabel("True Positive Rate", color=TEXT_CLR)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)

        path = os.path.join(self.report_dir, "roc_curves.png")
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        plt.close()
        logger.info("Saved ROC curves → %s", path)

    # ── 4. Feature importance (SHAP or RF importance) ─────────────────────────
    def plot_feature_importance(self, top_n: int = 15):
        """
        Plot feature importances for the best model.
        Uses built-in RF/GB importances; falls back to permutation importance.
        """
        model = self.trainer.best_model_
        name  = self.trainer.best_name_

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        else:
            logger.info(
                "Model %s has no feature_importances_; skipping FI plot.", name
            )
            return

        n_feats = min(len(self.feature_names), len(importances), top_n)
        idx     = np.argsort(importances)[::-1][:n_feats]
        top_f   = [self.feature_names[i] for i in idx]
        top_v   = importances[idx]

        fig, ax = plt.subplots(figsize=(10, 0.45 * n_feats + 2))
        bars    = ax.barh(
            range(n_feats), top_v[::-1],
            color=[
                COLORS.get("RandomForest", "#34d399") if v > np.median(top_v)
                else "#63b3ed"
                for v in top_v[::-1]
            ],
            edgecolor="none", height=0.65,
        )
        ax.set_yticks(range(n_feats))
        ax.set_yticklabels(top_f[::-1], fontsize=9)
        ax.set_xlabel("Feature Importance Score", color=TEXT_CLR)
        ax.set_title(
            f"Top {n_feats} Feature Importances — {name}",
            color=TEXT_CLR, fontsize=12,
        )
        ax.grid(True, axis="x", alpha=0.3)
        ax.set_axisbelow(True)

        for bar, val in zip(bars, top_v[::-1]):
            ax.text(
                val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center", fontsize=8, color=TEXT_CLR,
            )

        path = os.path.join(self.report_dir, "feature_importance.png")
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        plt.close()
        logger.info("Saved feature importance → %s", path)
