"""SageMaker inference handler for the F1 Race Position Predictor.

Uses the SKLearn container's hosting interface:
- model_fn: Load model from model directory
- input_fn: Deserialize request data
- predict_fn: Run prediction
- output_fn: Serialize response

NOTE: This file intentionally avoids `from __future__ import annotations`
and modern type hints for maximum compatibility with SageMaker containers.
"""

import json
import os
import logging
import pickle

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "grid_position",
    "rolling_avg_finish_short",
    "rolling_avg_finish_long",
    "rolling_avg_points",
    "position_delta_trend",
    "circuit_avg_finish",
    "circuit_race_count",
    "team_season_avg_finish",
    "team_points_per_race",
    "dnf_rate_season",
    "dnf_rate_circuit",
    "quali_position",
    "grid_quali_delta",
    "air_temperature",
    "track_temperature",
    "is_wet_race",
    "humidity",
    "wind_speed",
    "n_pit_stops",
    "is_street_circuit",
    "teammate_delta_rolling",
]

FEATURE_DEFAULTS = {
    "grid_position": 10.0,
    "rolling_avg_finish_short": 10.0,
    "rolling_avg_finish_long": 10.0,
    "rolling_avg_points": 5.0,
    "position_delta_trend": 0.0,
    "circuit_avg_finish": 10.0,
    "circuit_race_count": 2,
    "team_season_avg_finish": 10.0,
    "team_points_per_race": 5.0,
    "dnf_rate_season": 0.1,
    "dnf_rate_circuit": 0.1,
    "quali_position": 10.0,
    "grid_quali_delta": 0.0,
    "air_temperature": 25.0,
    "track_temperature": 40.0,
    "is_wet_race": 0,
    "humidity": 55.0,
    "wind_speed": 2.0,
    "n_pit_stops": 1,
    "is_street_circuit": 0,
    "teammate_delta_rolling": 0.0,
}


def model_fn(model_dir):
    """Load model from the SageMaker model directory.

    Uses pickle as fallback if joblib isn't available.
    """
    model_path = os.path.join(model_dir, "xgb_f1_model.joblib")
    quantile_path = os.path.join(model_dir, "quantile_models.joblib")

    try:
        import joblib
        model = joblib.load(model_path)
    except ImportError:
        with open(model_path, "rb") as f:
            model = pickle.load(f)

    logger.info("Loaded main model from %s", model_path)

    quantile_models = None
    if os.path.exists(quantile_path):
        try:
            import joblib
            quantile_models = joblib.load(quantile_path)
        except ImportError:
            with open(quantile_path, "rb") as f:
                quantile_models = pickle.load(f)
        logger.info("Loaded quantile models from %s", quantile_path)

    return {"model": model, "quantile": quantile_models}


def input_fn(request_body, request_content_type):
    """Deserialize input data to a DataFrame."""
    if request_content_type == "application/json":
        data = json.loads(request_body)

        if isinstance(data, dict):
            data = [data]

        filled = []
        for row in data:
            filled_row = {}
            for col in FEATURE_COLUMNS:
                filled_row[col] = row.get(col, FEATURE_DEFAULTS[col])
            filled.append(filled_row)

        return pd.DataFrame(filled)[FEATURE_COLUMNS]

    elif request_content_type == "text/csv":
        from io import StringIO
        df = pd.read_csv(StringIO(request_body))
        for col in FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = FEATURE_DEFAULTS[col]
        return df[FEATURE_COLUMNS]

    else:
        raise ValueError("Unsupported content type: %s" % request_content_type)


def predict_fn(input_data, model_dict):
    """Run prediction on the input data."""
    model = model_dict["model"]
    quantile = model_dict["quantile"]

    raw_preds = model.predict(input_data)
    predictions = np.clip(np.round(raw_preds), 1, 20).astype(int).tolist()

    result = {"predictions": predictions}

    if quantile is not None:
        lower = np.clip(np.round(quantile["lower"].predict(input_data)), 1, 20).astype(int).tolist()
        median = np.clip(np.round(quantile["median"].predict(input_data)), 1, 20).astype(int).tolist()
        upper = np.clip(np.round(quantile["upper"].predict(input_data)), 1, 20).astype(int).tolist()
        result["confidence_intervals"] = {
            "lower_10": lower,
            "median": median,
            "upper_90": upper,
        }

    return result


def output_fn(prediction, response_content_type):
    """Serialize prediction output."""
    return json.dumps(prediction)
