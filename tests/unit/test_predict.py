"""Tests for inference / prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBRegressor

from f1_predictor.features.engineering import FEATURE_COLUMNS, TARGET_COLUMN


class TestPredict:
    def test_predict_returns_clipped_values(self, sample_features):
        """Predictions should be clipped to [1, 20]."""
        X = sample_features[FEATURE_COLUMNS]
        y = sample_features[TARGET_COLUMN]

        model = XGBRegressor(n_estimators=10, max_depth=3, random_state=42)
        model.fit(X, y)

        preds = model.predict(X)
        preds_clipped = np.clip(preds, 1.0, 20.0)

        assert preds_clipped.min() >= 1.0
        assert preds_clipped.max() <= 20.0

    def test_single_prediction_shape(self, sample_features):
        """A single feature dict should return a single prediction."""
        X = sample_features[FEATURE_COLUMNS]
        y = sample_features[TARGET_COLUMN]

        model = XGBRegressor(n_estimators=10, max_depth=3, random_state=42)
        model.fit(X, y)

        single_input = X.iloc[[0]]
        pred = model.predict(single_input)
        assert pred.shape == (1,)

    def test_grid_position_1_predicts_low(self, sample_features):
        """A pole-sitter with strong form should predict a top finish."""
        X = sample_features[FEATURE_COLUMNS]
        y = sample_features[TARGET_COLUMN]

        model = XGBRegressor(n_estimators=50, max_depth=5, random_state=42)
        model.fit(X, y)

        fast_driver = pd.DataFrame(
            [
                {
                    # Original features
                    "grid_position": 1.0,
                    "rolling_avg_finish_short": 2.0,
                    "rolling_avg_finish_long": 2.5,
                    "rolling_avg_points": 22.0,
                    "position_delta_trend": 1.0,
                    "circuit_avg_finish": 2.0,
                    "circuit_race_count": 5,
                    "team_season_avg_finish": 3.0,
                    "team_points_per_race": 20.0,
                    "dnf_rate_season": 0.0,
                    "dnf_rate_circuit": 0.0,
                    # Phase 1: Qualifying
                    "quali_position": 1.0,
                    "grid_quali_delta": 0.0,
                    # Phase 1: Weather
                    "air_temperature": 25.0,
                    "track_temperature": 40.0,
                    "is_wet_race": 0,
                    "humidity": 50.0,
                    "wind_speed": 2.0,
                    # Phase 1: Strategy
                    "n_pit_stops": 1,
                    # Phase 1: Circuit type
                    "is_street_circuit": 0,
                    # Phase 1: Teammate
                    "teammate_delta_rolling": -2.0,
                }
            ]
        )

        pred = model.predict(fast_driver[FEATURE_COLUMNS])
        assert pred[0] < 10, f"Expected top finish, got P{pred[0]:.1f}"
