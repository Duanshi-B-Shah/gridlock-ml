"""Confidence intervals via quantile regression.

Trains 3 XGBoost models at different quantiles (10th, 50th, 90th percentile)
to produce prediction intervals: "Predicted P5 (range: P3–P7)".
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from f1_predictor.config import settings
from f1_predictor.features.engineering import FEATURE_COLUMNS, TARGET_COLUMN

logger = logging.getLogger(__name__)


class QuantilePredictor:
    """Predicts finishing positions with confidence intervals using quantile regression."""

    def __init__(
        self,
        lower_quantile: float = 0.1,
        upper_quantile: float = 0.9,
    ) -> None:
        """Initialize with quantile bounds.

        Args:
            lower_quantile: Lower bound percentile (default: 10th).
            upper_quantile: Upper bound percentile (default: 90th).
        """
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

        self.model_lower: XGBRegressor | None = None
        self.model_median: XGBRegressor | None = None
        self.model_upper: XGBRegressor | None = None

    def train(
        self,
        X: pd.DataFrame | None = None,
        y: pd.Series | None = None,
        n_estimators: int = 200,
        max_depth: int = 5,
        learning_rate: float = 0.1,
    ) -> None:
        """Train three quantile regression models.

        Args:
            X: Feature matrix. Loads from disk if None.
            y: Target vector. Loads from disk if None.
            n_estimators: Number of boosting rounds.
            max_depth: Max tree depth.
            learning_rate: Learning rate.
        """
        if X is None or y is None:
            features_path = settings.project_root / settings.data_processed_dir / "features.csv"
            df = pd.read_csv(features_path)
            X = df[FEATURE_COLUMNS]
            y = df[TARGET_COLUMN]

        common_params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
        }

        logger.info(f"Training lower quantile model (q={self.lower_quantile})")
        self.model_lower = XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=self.lower_quantile,
            **common_params,
        )
        self.model_lower.fit(X, y)

        logger.info("Training median model (q=0.5)")
        self.model_median = XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=0.5,
            **common_params,
        )
        self.model_median.fit(X, y)

        logger.info(f"Training upper quantile model (q={self.upper_quantile})")
        self.model_upper = XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=self.upper_quantile,
            **common_params,
        )
        self.model_upper.fit(X, y)

        logger.info("All 3 quantile models trained")

    def predict(self, features: dict[str, float] | pd.DataFrame) -> dict[str, np.ndarray]:
        """Predict with confidence intervals.

        Args:
            features: Single feature dict or DataFrame.

        Returns:
            Dict with 'lower', 'median', 'upper' arrays (all clipped to [1, 20]).
        """
        if self.model_lower is None:
            raise RuntimeError("Models not trained. Call .train() first.")

        if isinstance(features, dict):
            X = pd.DataFrame([features])[FEATURE_COLUMNS]
        else:
            X = features[FEATURE_COLUMNS]

        lower = np.clip(np.round(self.model_lower.predict(X)), 1, 20).astype(int)
        median = np.clip(np.round(self.model_median.predict(X)), 1, 20).astype(int)
        upper = np.clip(np.round(self.model_upper.predict(X)), 1, 20).astype(int)

        # Ensure lower <= median <= upper
        lower = np.minimum(lower, median)
        upper = np.maximum(upper, median)

        return {"lower": lower, "median": median, "upper": upper}

    def predict_single(self, features: dict[str, float]) -> dict[str, int]:
        """Predict a single driver with confidence interval.

        Returns:
            Dict with 'lower', 'median', 'upper' as ints.
        """
        result = self.predict(features)
        return {
            "lower": int(result["lower"][0]),
            "median": int(result["median"][0]),
            "upper": int(result["upper"][0]),
        }

    def save(self, path: Path | None = None) -> Path:
        """Save all 3 quantile models to disk.

        Returns:
            Path to saved file.
        """
        if path is None:
            settings.ensure_dirs()
            path = settings.project_root / settings.models_dir / "quantile_models.joblib"

        joblib.dump(
            {
                "lower": self.model_lower,
                "median": self.model_median,
                "upper": self.model_upper,
                "lower_quantile": self.lower_quantile,
                "upper_quantile": self.upper_quantile,
            },
            path,
        )
        logger.info(f"Quantile models saved to {path}")
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> QuantilePredictor:
        """Load quantile models from disk.

        Returns:
            Initialized QuantilePredictor with loaded models.
        """
        if path is None:
            path = settings.project_root / settings.models_dir / "quantile_models.joblib"

        if not path.exists():
            raise FileNotFoundError(
                f"Quantile models not found at {path}. "
                "Run `python scripts/train_quantile.py` first."
            )

        data = joblib.load(path)

        predictor = cls(
            lower_quantile=data["lower_quantile"],
            upper_quantile=data["upper_quantile"],
        )
        predictor.model_lower = data["lower"]
        predictor.model_median = data["median"]
        predictor.model_upper = data["upper"]

        logger.info(f"Quantile models loaded from {path}")
        return predictor
