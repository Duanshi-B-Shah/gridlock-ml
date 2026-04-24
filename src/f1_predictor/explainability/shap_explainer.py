"""SHAP-based explainability for the F1 position predictor.

Provides:
- Global feature importance (beeswarm / summary plots)
- Per-prediction explanations (waterfall charts)
- SHAP value computation with caching
"""

from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from f1_predictor.config import settings
from f1_predictor.features.engineering import FEATURE_COLUMNS, TARGET_COLUMN

logger = logging.getLogger(__name__)


class F1Explainer:
    """SHAP explainer for the F1 race position model."""

    def __init__(self, model=None, X: pd.DataFrame | None = None) -> None:
        """Initialize the explainer.

        Args:
            model: Trained XGBoost model. Loads from disk if None.
            X: Feature matrix for background data. Loads from disk if None.
        """
        if model is None:
            model = joblib.load(settings.model_path)
        self.model = model

        if X is None:
            features_path = settings.project_root / settings.data_processed_dir / "features.csv"
            df = pd.read_csv(features_path)
            X = df[FEATURE_COLUMNS]
        self.X = X

        # TreeExplainer is exact and fast for XGBoost
        self.explainer = shap.TreeExplainer(self.model)
        self._shap_values = None

        logger.info("F1Explainer initialized with TreeExplainer")

    @property
    def shap_values(self) -> shap.Explanation:
        """Compute SHAP values for the full dataset (cached)."""
        if self._shap_values is None:
            logger.info(f"Computing SHAP values for {len(self.X)} samples...")
            self._shap_values = self.explainer(self.X)
            logger.info("SHAP values computed")
        return self._shap_values

    def explain_single(self, idx: int) -> shap.Explanation:
        """Get SHAP explanation for a single prediction by index.

        Args:
            idx: Row index in the dataset.

        Returns:
            SHAP Explanation object for that prediction.
        """
        return self.shap_values[idx]

    def explain_features(self, features: dict[str, float]) -> shap.Explanation:
        """Get SHAP explanation for a custom feature dict.

        Args:
            features: Dict of feature name -> value.

        Returns:
            SHAP Explanation object.
        """
        X_single = pd.DataFrame([features])[FEATURE_COLUMNS]
        return self.explainer(X_single)[0]

    def get_top_contributors(
        self, idx: int, n: int = 5
    ) -> list[tuple[str, float, float]]:
        """Get top N features contributing to a prediction.

        Args:
            idx: Row index in the dataset.
            n: Number of top features to return.

        Returns:
            List of (feature_name, shap_value, feature_value) tuples,
            sorted by absolute SHAP impact.
        """
        sv = self.shap_values[idx]
        abs_vals = np.abs(sv.values)
        top_indices = np.argsort(abs_vals)[-n:][::-1]

        return [
            (FEATURE_COLUMNS[i], float(sv.values[i]), float(sv.data[i]))
            for i in top_indices
        ]

    # ─── Plot generation ───

    def plot_waterfall(
        self,
        idx: int | None = None,
        features: dict[str, float] | None = None,
        save_path: Path | None = None,
        max_display: int = 15,
    ) -> Path | None:
        """Generate a SHAP waterfall plot for a single prediction.

        Args:
            idx: Row index in the dataset. Used if features is None.
            features: Custom feature dict. Takes priority over idx.
            save_path: Where to save the plot. Auto-generates if None.
            max_display: Max features to display.

        Returns:
            Path to saved plot, or None if not saved.
        """
        if features is not None:
            explanation = self.explain_features(features)
        elif idx is not None:
            explanation = self.explain_single(idx)
        else:
            raise ValueError("Either idx or features must be provided")

        fig = plt.figure(figsize=(12, 8))
        shap.plots.waterfall(explanation, max_display=max_display, show=False)
        plt.title("SHAP Waterfall — Why This Prediction?", fontsize=14)
        plt.tight_layout()

        if save_path is None:
            save_dir = settings.project_root / settings.plots_dir
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / "shap_waterfall.png"

        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Waterfall plot saved to {save_path}")
        return save_path

    def plot_beeswarm(self, save_path: Path | None = None, max_display: int = 20) -> Path:
        """Generate a SHAP beeswarm (global importance) plot.

        Args:
            save_path: Where to save the plot.
            max_display: Max features to display.

        Returns:
            Path to saved plot.
        """
        fig = plt.figure(figsize=(12, 8))
        shap.plots.beeswarm(self.shap_values, max_display=max_display, show=False)
        plt.title("SHAP Beeswarm — Global Feature Impact", fontsize=14)
        plt.tight_layout()

        if save_path is None:
            save_dir = settings.project_root / settings.plots_dir
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / "shap_beeswarm.png"

        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Beeswarm plot saved to {save_path}")
        return save_path

    def plot_bar(self, save_path: Path | None = None, max_display: int = 20) -> Path:
        """Generate a SHAP bar plot (mean absolute SHAP values).

        Args:
            save_path: Where to save the plot.
            max_display: Max features to display.

        Returns:
            Path to saved plot.
        """
        fig = plt.figure(figsize=(12, 8))
        shap.plots.bar(self.shap_values, max_display=max_display, show=False)
        plt.title("SHAP Feature Importance — Mean |SHAP Value|", fontsize=14)
        plt.tight_layout()

        if save_path is None:
            save_dir = settings.project_root / settings.plots_dir
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / "shap_bar.png"

        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Bar plot saved to {save_path}")
        return save_path

    def get_waterfall_figure(
        self,
        idx: int | None = None,
        features: dict[str, float] | None = None,
        max_display: int = 15,
    ) -> plt.Figure:
        """Get a waterfall plot as a matplotlib Figure (for Streamlit).

        Returns:
            matplotlib Figure object.
        """
        if features is not None:
            explanation = self.explain_features(features)
        elif idx is not None:
            explanation = self.explain_single(idx)
        else:
            raise ValueError("Either idx or features must be provided")

        fig = plt.figure(figsize=(12, 8))
        shap.plots.waterfall(explanation, max_display=max_display, show=False)
        plt.tight_layout()
        return fig

    def get_beeswarm_figure(self, max_display: int = 20) -> plt.Figure:
        """Get a beeswarm plot as a matplotlib Figure (for Streamlit).

        Returns:
            matplotlib Figure object.
        """
        fig = plt.figure(figsize=(12, 8))
        shap.plots.beeswarm(self.shap_values, max_display=max_display, show=False)
        plt.tight_layout()
        return fig
