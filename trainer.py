"""
trainer.py — Train and cross-validate all ML classifiers.

Implements:
  - Stratified K-Fold cross-validation (10 folds)
  - SMOTE applied inside each training fold (no data leakage)
  - All 6 classifiers from the base paper
  - Full metric collection: Accuracy, Precision, Recall, F1, MCC, AUC, FPR
  - Saves the best model to disk

Usage:
    trainer = ModelTrainer(config)
    results = trainer.run(X, y)
    trainer.save_best_model()
"""

import os
import time
import logging
import pickle
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
)

from imblearn.over_sampling import SMOTE

logger = logging.getLogger(__name__)


def _build_classifier(name: str, params: dict):
    """Instantiate a classifier by name with given hyperparameters."""
    mapping = {
        "RandomForest":      RandomForestClassifier,
        "GradientBoosting":  GradientBoostingClassifier,
        "AdaBoost":          AdaBoostClassifier,
        "SVM":               SVC,
        "KNN":               KNeighborsClassifier,
        "NaiveBayes":        GaussianNB,
    }
    if name not in mapping:
        raise ValueError(f"Unknown classifier: {name}. Choose from {list(mapping)}")
    return mapping[name](**params)


class ModelTrainer:
    """
    Trains, validates, and compares all specified ML classifiers.

    Parameters
    ----------
    config : module
        The project config module (config.py).
    """

    def __init__(self, config):
        self.config       = config
        self.results_     = {}    # {clf_name: {metric: [fold_scores]}}
        self.best_model_  = None
        self.best_name_   = None
        self.best_acc_    = 0.0

    # ── main entry ────────────────────────────────────────────────────────────
    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> pd.DataFrame:
        """
        Run 10-fold CV for all classifiers and return a summary DataFrame.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)
        """
        cfg    = self.config
        skf    = StratifiedKFold(
            n_splits=cfg.N_CV_FOLDS, shuffle=True, random_state=cfg.RANDOM_STATE
        )
        smote  = SMOTE(sampling_strategy=cfg.SMOTE_RATIO, random_state=cfg.RANDOM_STATE)

        for clf_name in cfg.CLASSIFIERS:
            logger.info("=" * 60)
            logger.info("Training: %s", clf_name)
            params  = cfg.HYPERPARAMS[clf_name]
            metrics = {
                "accuracy": [], "precision": [], "recall": [],
                "f1": [], "mcc": [], "auc": [], "fpr": [],
            }
            fold_cms = []

            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
                X_tr, X_te = X[train_idx], X[test_idx]
                y_tr, y_te = y[train_idx], y[test_idx]

                # SMOTE on training fold only
                if cfg.USE_SMOTE:
                    X_tr, y_tr = smote.fit_resample(X_tr, y_tr)

                clf = _build_classifier(clf_name, params)
                t0  = time.time()
                clf.fit(X_tr, y_tr)
                elapsed = time.time() - t0

                y_pred  = clf.predict(X_te)
                y_proba = (
                    clf.predict_proba(X_te)[:, 1]
                    if hasattr(clf, "predict_proba")
                    else clf.decision_function(X_te)
                )

                # Confusion matrix
                tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
                fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0

                metrics["accuracy"].append(accuracy_score(y_te, y_pred))
                metrics["precision"].append(
                    precision_score(y_te, y_pred, zero_division=0)
                )
                metrics["recall"].append(recall_score(y_te, y_pred, zero_division=0))
                metrics["f1"].append(f1_score(y_te, y_pred, zero_division=0))
                metrics["mcc"].append(matthews_corrcoef(y_te, y_pred))
                metrics["auc"].append(roc_auc_score(y_te, y_proba))
                metrics["fpr"].append(fpr_val)
                fold_cms.append((tn, fp, fn, tp))

                logger.info(
                    "  Fold %2d | Acc=%.4f | F1=%.4f | AUC=%.4f | %.1fs",
                    fold_idx,
                    metrics["accuracy"][-1],
                    metrics["f1"][-1],
                    metrics["auc"][-1],
                    elapsed,
                )

            self.results_[clf_name] = {
                k: np.array(v) for k, v in metrics.items()
            }
            self.results_[clf_name]["confusion_matrices"] = fold_cms

            mean_acc = np.mean(metrics["accuracy"])
            if mean_acc > self.best_acc_:
                self.best_acc_  = mean_acc
                self.best_name_ = clf_name
                # Re-fit best model on full training data
                X_full, y_full = (
                    smote.fit_resample(X, y) if cfg.USE_SMOTE else (X, y)
                )
                best_clf = _build_classifier(clf_name, params)
                best_clf.fit(X_full, y_full)
                self.best_model_ = best_clf

            logger.info(
                "  Mean Accuracy: %.4f | Best so far: %s (%.4f)",
                mean_acc, self.best_name_, self.best_acc_,
            )

        logger.info("=" * 60)
        logger.info("Best model: %s  Accuracy=%.4f", self.best_name_, self.best_acc_)
        return self.summary()

    # ── summary DataFrame ─────────────────────────────────────────────────────
    def summary(self) -> pd.DataFrame:
        """Return a DataFrame with mean ± std for each metric per classifier."""
        rows = []
        for name, metrics in self.results_.items():
            row = {"Algorithm": name}
            for metric, values in metrics.items():
                if metric == "confusion_matrices":
                    continue
                row[f"{metric}_mean"] = np.mean(values)
                row[f"{metric}_std"]  = np.std(values)
            rows.append(row)
        return pd.DataFrame(rows).sort_values("accuracy_mean", ascending=False)

    # ── save / load ───────────────────────────────────────────────────────────
    def save_best_model(self, model_dir: str):
        """Pickle the best trained model to model_dir."""
        os.makedirs(model_dir, exist_ok=True)
        path = os.path.join(model_dir, f"{self.best_name_}_best.pkl")
        with open(path, "wb") as f:
            pickle.dump(self.best_model_, f)
        logger.info("Saved best model (%s) → %s", self.best_name_, path)
        return path

    @staticmethod
    def load_model(model_path: str):
        """Load a pickled model from disk."""
        with open(model_path, "rb") as f:
            return pickle.load(f)
