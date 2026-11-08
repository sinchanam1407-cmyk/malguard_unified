"""
feature_selector.py — Information Gain-based feature selection and preprocessing.

Steps:
  1. Drop near-zero variance features
  2. Rank remaining features by Information Gain (mutual_info_classif)
  3. Retain top-K features
  4. Optionally remove highly correlated features (Pearson r > threshold)
  5. Return selected feature names + fitted selector (for transform at test time)

Usage:
    selector = FeatureSelector(top_k=100, corr_threshold=0.95)
    X_sel    = selector.fit_transform(X_train, y_train)
    X_test_s = selector.transform(X_test)
    print(selector.selected_features_)
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Optional

from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


class FeatureSelector:
    """
    Information Gain-based feature selection with optional correlation filter.

    Parameters
    ----------
    top_k : int
        Number of top-ranked features to retain.
    corr_threshold : float
        Pearson correlation threshold; one of each correlated pair is dropped.
        Set to 1.0 to disable.
    variance_threshold : float
        Features with variance < this are dropped first.
    scale : bool
        If True, apply MinMaxScaler to the selected features.
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        top_k: int = 100,
        corr_threshold: float = 0.95,
        variance_threshold: float = 0.005,
        scale: bool = True,
        random_state: int = 42,
    ):
        self.top_k              = top_k
        self.corr_threshold     = corr_threshold
        self.variance_threshold = variance_threshold
        self.scale              = scale
        self.random_state       = random_state

        self.selected_features_: List[str] = []
        self.ig_scores_: Optional[np.ndarray] = None
        self._scaler: Optional[MinMaxScaler] = None

    # ── fit_transform ─────────────────────────────────────────────────────────
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit the selector on (X, y) and return transformed X."""
        feature_names = list(X.columns)

        # Step 1 – Drop near-zero variance features
        variances = X.var()
        keep_var  = variances[variances >= self.variance_threshold].index.tolist()
        logger.info(
            "Variance filter: %d → %d features", len(feature_names), len(keep_var)
        )
        X_var = X[keep_var]

        # Step 2 – Information Gain ranking
        ig_scores = mutual_info_classif(
            X_var, y, discrete_features=False, random_state=self.random_state
        )
        self.ig_scores_ = ig_scores

        ig_series = pd.Series(ig_scores, index=keep_var).sort_values(ascending=False)
        top_features = ig_series.head(self.top_k).index.tolist()
        logger.info(
            "IG selection: %d → %d features (top-%d)",
            len(keep_var), len(top_features), self.top_k,
        )

        # Step 3 – Correlation filter
        if self.corr_threshold < 1.0:
            top_features = self._remove_correlated(X_var[top_features])
            logger.info(
                "Correlation filter: retained %d features (threshold=%.2f)",
                len(top_features), self.corr_threshold,
            )

        self.selected_features_ = top_features
        X_sel = X_var[top_features]

        # Step 4 – Scale
        if self.scale:
            self._scaler = MinMaxScaler()
            X_scaled = pd.DataFrame(
                self._scaler.fit_transform(X_sel),
                columns=top_features,
                index=X.index,
            )
            return X_scaled

        return X_sel

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using the fitted selector."""
        missing = [f for f in self.selected_features_ if f not in X.columns]
        if missing:
            logger.warning("Missing %d features in test set; filling with 0", len(missing))
            for f in missing:
                X[f] = 0

        X_sel = X[self.selected_features_]
        if self.scale and self._scaler:
            return pd.DataFrame(
                self._scaler.transform(X_sel),
                columns=self.selected_features_,
                index=X.index,
            )
        return X_sel

    # ── correlation filter ────────────────────────────────────────────────────
    def _remove_correlated(self, X: pd.DataFrame) -> List[str]:
        """Return feature list with highly correlated pairs removed."""
        corr_matrix = X.corr().abs()
        upper       = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop = [
            col for col in upper.columns
            if any(upper[col] > self.corr_threshold)
        ]
        return [c for c in X.columns if c not in to_drop]

    # ── utilities ─────────────────────────────────────────────────────────────
    def get_ig_report(self) -> pd.DataFrame:
        """Return a DataFrame of all features with their IG scores."""
        if self.ig_scores_ is None:
            raise RuntimeError("Call fit_transform() first.")
        # Reconstruct full report from selected features + scores
        return pd.DataFrame(
            {"feature": self.selected_features_}
        )
