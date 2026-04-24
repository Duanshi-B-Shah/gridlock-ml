"""Tests for feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1_predictor.features.engineering import (
    FEATURE_COLUMNS,
    STREET_CIRCUITS,
    TARGET_COLUMN,
    _assign_points,
    build_features,
)


class TestAssignPoints:
    def test_winner_gets_25(self):
        assert _assign_points(1) == 25.0

    def test_second_gets_18(self):
        assert _assign_points(2) == 18.0

    def test_tenth_gets_1(self):
        assert _assign_points(10) == 1.0

    def test_eleventh_gets_zero(self):
        assert _assign_points(11) == 0.0

    def test_nan_gets_zero(self):
        assert _assign_points(float("nan")) == 0.0

    def test_none_gets_zero(self):
        assert _assign_points(None) == 0.0


class TestBuildFeatures:
    def test_output_has_all_feature_columns(self, sample_raw_data, tmp_path):
        csv_path = tmp_path / "raw.csv"
        sample_raw_data.to_csv(csv_path, index=False)

        result = build_features([csv_path])

        for col in FEATURE_COLUMNS:
            assert col in result.columns, f"Missing feature column: {col}"
        assert TARGET_COLUMN in result.columns

    def test_no_nan_in_features(self, sample_raw_data, tmp_path):
        csv_path = tmp_path / "raw.csv"
        sample_raw_data.to_csv(csv_path, index=False)

        result = build_features([csv_path])

        for col in FEATURE_COLUMNS:
            assert result[col].isna().sum() == 0, f"NaN found in {col}"

    def test_grid_position_in_range(self, sample_raw_data, tmp_path):
        csv_path = tmp_path / "raw.csv"
        sample_raw_data.to_csv(csv_path, index=False)

        result = build_features([csv_path])

        assert result["grid_position"].min() >= 1
        assert result["grid_position"].max() <= 20


class TestPhase1Features:
    """Tests for Phase 1: Richer Data & Features."""

    def test_street_circuit_detection(self, sample_raw_data, tmp_path):
        """Monaco should be flagged as street circuit, Monza should not."""
        csv_path = tmp_path / "raw.csv"
        sample_raw_data.to_csv(csv_path, index=False)

        result = build_features([csv_path])

        monaco_rows = result[result["circuit_short_name"] == "Monaco"]
        monza_rows = result[result["circuit_short_name"] == "Monza"]

        assert (monaco_rows["is_street_circuit"] == 1).all(), "Monaco should be a street circuit"
        assert (monza_rows["is_street_circuit"] == 0).all(), "Monza should NOT be a street circuit"

    def test_grid_quali_delta_computation(self, sample_raw_data, tmp_path):
        """grid_quali_delta = grid_position - quali_position."""
        csv_path = tmp_path / "raw.csv"
        sample_raw_data.to_csv(csv_path, index=False)

        result = build_features([csv_path])

        expected = result["grid_position"] - result["quali_position"]
        pd.testing.assert_series_equal(
            result["grid_quali_delta"],
            expected,
            check_names=False,
        )

    def test_weather_features_present(self, sample_raw_data, tmp_path):
        """Weather features should exist and have no NaN."""
        csv_path = tmp_path / "raw.csv"
        sample_raw_data.to_csv(csv_path, index=False)

        result = build_features([csv_path])

        weather_cols = ["air_temperature", "track_temperature", "humidity", "wind_speed", "is_wet_race"]
        for col in weather_cols:
            assert col in result.columns, f"Missing weather column: {col}"
            assert result[col].isna().sum() == 0, f"NaN in {col}"

    def test_wet_race_flag(self, sample_raw_data, tmp_path):
        """Monaco race should be flagged as wet."""
        csv_path = tmp_path / "raw.csv"
        sample_raw_data.to_csv(csv_path, index=False)

        result = build_features([csv_path])

        monaco_rows = result[result["circuit_short_name"] == "Monaco"]
        assert (monaco_rows["is_wet_race"] == 1).all(), "Monaco should be wet"

    def test_pit_stops_feature(self, sample_raw_data, tmp_path):
        """n_pit_stops should be present and non-negative."""
        csv_path = tmp_path / "raw.csv"
        sample_raw_data.to_csv(csv_path, index=False)

        result = build_features([csv_path])

        assert "n_pit_stops" in result.columns
        assert (result["n_pit_stops"] >= 0).all()

    def test_teammate_delta_no_leakage(self, sample_raw_data, tmp_path):
        """Teammate delta should use shift(1) - no NaN in output."""
        csv_path = tmp_path / "raw.csv"
        sample_raw_data.to_csv(csv_path, index=False)

        result = build_features([csv_path])

        assert "teammate_delta_rolling" in result.columns
        assert result["teammate_delta_rolling"].isna().sum() == 0

    def test_backward_compatibility_no_new_columns(self, tmp_path):
        """Old raw data without Phase 1 columns should still work."""
        np.random.seed(42)
        n = 20
        old_data = pd.DataFrame({
            "session_key": [100] * n,
            "driver_number": list(range(1, n + 1)),
            "full_name": [f"Driver {i}" for i in range(1, n + 1)],
            "team_name": [f"Team {(i - 1) // 2 + 1}" for i in range(1, n + 1)],
            "finishing_position": np.random.randint(1, 21, n),
            "grid_position": np.random.randint(1, 21, n),
            "circuit_short_name": ["Monza"] * n,
            "year": [2024] * n,
        })

        csv_path = tmp_path / "old_raw.csv"
        old_data.to_csv(csv_path, index=False)

        result = build_features([csv_path])

        for col in FEATURE_COLUMNS:
            assert col in result.columns, f"Missing column in backward-compatible mode: {col}"
            assert result[col].isna().sum() == 0, f"NaN in {col} with old data"
