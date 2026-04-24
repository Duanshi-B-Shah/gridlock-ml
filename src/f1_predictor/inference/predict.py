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
        Array of predicted finishing positions (integers, clipped to [1, 20]).
    """
    model = load_model()

    if isinstance(features, dict):
        df = pd.DataFrame([features])
    else:
        df = features

    # Ensure correct column order
    X = df[FEATURE_COLUMNS]

    predictions = model.predict(X)

    # Clip to valid range [1, 20] and round to whole positions
    # F1 positions are always integers (P1, P2, ... P20)
    predictions = np.clip(predictions, 1.0, 20.0)
    predictions = np.round(predictions).astype(int)

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
    # Phase 1 features
    quali_position: float = 10.0,
    grid_quali_delta: float = 0.0,
    air_temperature: float = 25.0,
    track_temperature: float = 40.0,
    is_wet_race: int = 0,
    humidity: float = 55.0,
    wind_speed: float = 2.0,
    n_pit_stops: int = 1,
    is_street_circuit: int = 0,
    teammate_delta_rolling: float = 0.0,
) -> int:
    """Predict a single driver's finishing position with explicit parameters.

    Returns:
        Predicted finishing position (int, P1-P20).
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
        "quali_position": quali_position,
        "grid_quali_delta": grid_quali_delta,
        "air_temperature": air_temperature,
        "track_temperature": track_temperature,
        "is_wet_race": is_wet_race,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "n_pit_stops": n_pit_stops,
        "is_street_circuit": is_street_circuit,
        "teammate_delta_rolling": teammate_delta_rolling,
    }
    return int(predict(features)[0])
