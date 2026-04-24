"""Shared test fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1_predictor.features.engineering import FEATURE_COLUMNS, TARGET_COLUMN


@pytest.fixture
def sample_features() -> pd.DataFrame:
    """Generate a small synthetic feature dataset with all 23 features."""
    np.random.seed(42)
    n = 50
    data = {
        # Original features
        "grid_position": np.random.randint(1, 21, n).astype(float),
        "rolling_avg_finish_short": np.random.uniform(2, 18, n),
        "rolling_avg_finish_long": np.random.uniform(3, 17, n),
        "rolling_avg_points": np.random.uniform(0, 25, n),
        "position_delta_trend": np.random.uniform(-5, 5, n),
        "circuit_avg_finish": np.random.uniform(3, 18, n),
        "circuit_race_count": np.random.randint(0, 8, n),
        "team_season_avg_finish": np.random.uniform(3, 18, n),
        "team_points_per_race": np.random.uniform(0, 25, n),
        "dnf_rate_season": np.random.uniform(0, 0.4, n),
        "dnf_rate_circuit": np.random.uniform(0, 0.5, n),
        # Phase 1: Qualifying
        "quali_position": np.random.randint(1, 21, n).astype(float),
        "grid_quali_delta": np.random.randint(-3, 4, n).astype(float),
        # Phase 1: Weather
        "air_temperature": np.random.uniform(15, 40, n),
        "track_temperature": np.random.uniform(20, 55, n),
        "is_wet_race": np.random.choice([0, 1], n, p=[0.8, 0.2]),
        "humidity": np.random.uniform(30, 95, n),
        "wind_speed": np.random.uniform(0, 8, n),
        # Phase 1: Strategy
        "n_pit_stops": np.random.choice([1, 2, 3], n, p=[0.6, 0.35, 0.05]),
        # Phase 1: Circuit type
        "is_street_circuit": np.random.choice([0, 1], n, p=[0.7, 0.3]),
        # Phase 1: Teammate comparison
        "teammate_delta_rolling": np.random.uniform(-5, 5, n),
    }
    df = pd.DataFrame(data)
    df[TARGET_COLUMN] = np.clip(
        df["grid_position"] + np.random.normal(0, 3, n), 1, 20
    )
    return df


@pytest.fixture
def sample_raw_data() -> pd.DataFrame:
    """Simulate raw race data as fetched from OpenF1 (including Phase 1 columns)."""
    np.random.seed(42)
    drivers = list(range(1, 21))
    sessions = [100, 101, 102, 103, 104]
    circuits = ["Monza", "Silverstone", "Monaco", "Spa", "Suzuka"]

    rows = []
    for i, sk in enumerate(sessions):
        for dn in drivers:
            grid_pos = np.random.randint(1, 21)
            quali_pos = max(1, min(20, grid_pos + np.random.choice([-1, 0, 0, 0, 1, 2])))
            rows.append(
                {
                    "session_key": sk,
                    "driver_number": dn,
                    "full_name": f"Driver {dn}",
                    "team_name": f"Team {(dn - 1) // 2 + 1}",
                    "finishing_position": np.random.randint(1, 21),
                    "grid_position": grid_pos,
                    "circuit_short_name": circuits[i],
                    "year": 2024,
                    "is_dnf": 1 if np.random.random() < 0.1 else 0,
                    # Weather (session-level)
                    "air_temperature": 25.0 + i * 2,
                    "track_temperature": 35.0 + i * 3,
                    "humidity": 60.0 + i * 5,
                    "wind_speed": 1.5 + i * 0.3,
                    "is_wet_race": 1 if i == 2 else 0,
                    # Stints
                    "n_pit_stops": np.random.choice([1, 2]),
                    "primary_compound": np.random.choice(["SOFT", "MEDIUM", "HARD"]),
                    "total_laps": np.random.randint(50, 70),
                    # Qualifying
                    "quali_position": quali_pos,
                }
            )
    return pd.DataFrame(rows)
