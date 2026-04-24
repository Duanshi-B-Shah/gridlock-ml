"""Model inference - load a saved model and predict finishing positions."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from f1_predictor.config import settings
from f1_predictor.features.engineering import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

_model_cache: XGBRegressor | None = None


def load_model(path: Path | None = None) -> XGBRegressor:
    """Load the trained model from disk (cached after first load).

    Args:
        path: Path to .joblib file. Defaults to config.

    Returns:
        Trained XGBRegressor.
    """
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    if path is None:
        path = settings.model_path

    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run `python scripts/train_model.py` first."
        )

    _model_cache = joblib.load(path)
    logger.info(f"Model loaded from {path}")
    return _model_cache


def predict(features: dict[str, float] | pd.DataFrame) -> np.ndarray:
    """Predict finishing position(s).

    Args:
        features: Either a single dict of feature values or a DataFrame.

    Returns:
        Array of predicted finishing positions.
    """
    model = load_model()

    if isinstance(features, dict):
        df = pd.DataFrame([features])
    else:
        df = features

    # Ensure correct column order
    X = df[FEATURE_COLUMNS]

    predictions = model.predict(X)

    # Clip to valid range [1, 20]
    predictions = np.clip(predictions, 1.0, 20.0)

    return predictions


def predict_single(
    grid_position: float,
    rolling_avg_finish_short: float = 10.0,
    rolling_avg_finish_long: float = 10.0,
    rolling_avg_points: float = 5.0,
    position_delta_trend: float = 0.0,
    circuit_avg_finish: float = 10.0,
    circuit_race_count: int = 2,
    team_season_avg_finish: float = 10.0,
    team_points_per_race: float = 5.0,
    dnf_rate_season: float = 0.1,
    dnf_rate_circuit: float = 0.1,
) -> float:
    """Predict a single driver's finishing position with explicit parameters.

    Returns:
        Predicted finishing position (float).
    """
    features = {
        "grid_position": grid_position,
        "rolling_avg_finish_short": rolling_avg_finish_short,
        "rolling_avg_finish_long": rolling_avg_finish_long,
        "rolling_avg_points": rolling_avg_points,
        "position_delta_trend": position_delta_trend,
        "circuit_avg_finish": circuit_avg_finish,
        "circuit_race_count": circuit_race_count,
        "team_season_avg_finish": team_season_avg_finish,
        "team_points_per_race": team_points_per_race,
        "dnf_rate_season": dnf_rate_season,
        "dnf_rate_circuit": dnf_rate_circuit,
    }
    return float(predict(features)[0])
