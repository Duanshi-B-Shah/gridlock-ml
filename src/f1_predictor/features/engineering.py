"""Feature engineering - transforms raw race results into model-ready features."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from f1_predictor.config import settings

logger = logging.getLogger(__name__)

# Street circuits behave very differently from permanent tracks
STREET_CIRCUITS = {"Monaco", "Jeddah", "Singapore", "Baku", "Las Vegas", "Melbourne", "Miami"}

FEATURE_COLUMNS = [
    # Original features
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
    # Phase 1: Qualifying
    "quali_position",
    "grid_quali_delta",
    # Phase 1: Weather
    "air_temperature",
    "track_temperature",
    "is_wet_race",
    "humidity",
    "wind_speed",
    # Phase 1: Strategy
    "n_pit_stops",
    # Phase 1: Circuit type
    "is_street_circuit",
    # Phase 1: Teammate comparison
    "teammate_delta_rolling",
]

TARGET_COLUMN = "finishing_position"


def _compute_dnf(df: pd.DataFrame) -> pd.DataFrame:
    """Flag DNFs - if finishing_position is NaN or driver didn't complete."""
    if "is_dnf" not in df.columns:
        df["is_dnf"] = df["finishing_position"].isna().astype(int)
    return df


def _assign_points(position: float | None) -> float:
    """F1 points system (top 10)."""
    points_map = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
    if position is None or np.isnan(position):
        return 0.0
    return points_map.get(int(position), 0.0)


def _build_qualifying_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build qualifying-derived features."""
    if "quali_position" not in df.columns:
        df["quali_position"] = df["grid_position"]
    else:
        df["quali_position"] = df["quali_position"].fillna(df["grid_position"])

    df["quali_position"] = pd.to_numeric(df["quali_position"], errors="coerce").fillna(df["grid_position"])
    df["grid_quali_delta"] = df["grid_position"] - df["quali_position"]

    return df


def _build_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure weather columns exist and fill missing values with medians."""
    weather_cols = {
        "air_temperature": None,
        "track_temperature": None,
        "humidity": None,
        "wind_speed": None,
        "is_wet_race": 0,
    }

    for col, default in weather_cols.items():
        if col not in df.columns:
            df[col] = default if default is not None else 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["air_temperature", "track_temperature", "humidity", "wind_speed"]:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val if not np.isnan(median_val) else 0.0)

    df["is_wet_race"] = df["is_wet_race"].fillna(0).astype(int)

    return df


def _build_stint_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure stint columns exist and fill missing values."""
    if "n_pit_stops" not in df.columns:
        df["n_pit_stops"] = 1
    else:
        df["n_pit_stops"] = pd.to_numeric(df["n_pit_stops"], errors="coerce").fillna(1).astype(int)

    return df


def _build_circuit_type_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Derive is_street_circuit from circuit_short_name."""
    if "circuit_short_name" in df.columns:
        df["is_street_circuit"] = df["circuit_short_name"].isin(STREET_CIRCUITS).astype(int)
    else:
        df["is_street_circuit"] = 0

    return df


def _build_teammate_delta(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling teammate finish delta. Uses .shift(1) to avoid leakage."""
    if "team_name" not in df.columns or "session_key" not in df.columns:
        df["teammate_delta_rolling"] = 0.0
        return df

    df["_teammate_finish"] = np.nan

    for (sk, team), group in df.groupby(["session_key", "team_name"]):
        if len(group) == 2:
            idx = group.index.tolist()
            df.loc[idx[0], "_teammate_finish"] = df.loc[idx[1], "finishing_position"]
            df.loc[idx[1], "_teammate_finish"] = df.loc[idx[0], "finishing_position"]

    df["_teammate_delta"] = df["finishing_position"] - df["_teammate_finish"]

    df["teammate_delta_rolling"] = df.groupby("driver_number")["_teammate_delta"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )

    df = df.drop(columns=["_teammate_finish", "_teammate_delta"], errors="ignore")
    df["teammate_delta_rolling"] = df["teammate_delta_rolling"].fillna(0.0)

    return df


def build_features(raw_csv_paths: list[Path]) -> pd.DataFrame:
    """Build feature-engineered dataset from raw race result CSVs.

    Args:
        raw_csv_paths: List of paths to raw CSV files (one per season).

    Returns:
        DataFrame with all features and target column, ready for training.
    """
    logger.info(f"Building features from {len(raw_csv_paths)} raw file(s)")

    frames = [pd.read_csv(p) for p in raw_csv_paths]
    df = pd.concat(frames, ignore_index=True)

    # -- Preprocessing --
    if "finishing_position" not in df.columns and "position" in df.columns:
        df = df.rename(columns={"position": "finishing_position"})

    if "grid_position" not in df.columns:
        logger.warning("No 'grid_position' column found — defaulting to 20 (back of grid)")
        df["grid_position"] = 20.0

    df = _compute_dnf(df)
    df["grid_position"] = pd.to_numeric(df["grid_position"], errors="coerce").fillna(20.0)

    df["points"] = df["finishing_position"].apply(_assign_points)

    df = df.dropna(subset=["finishing_position"])
    df["finishing_position"] = df["finishing_position"].astype(float)

    if "date_start" in df.columns:
        df = df.sort_values(["date_start", "driver_number"])
    else:
        df = df.sort_values(["session_key", "driver_number"])

    short_w = settings.rolling_window_short
    long_w = settings.rolling_window_long

    # -- Driver rolling averages --
    driver_group = df.groupby("driver_number")

    df["rolling_avg_finish_short"] = driver_group["finishing_position"].transform(
        lambda x: x.shift(1).rolling(short_w, min_periods=1).mean()
    )
    df["rolling_avg_finish_long"] = driver_group["finishing_position"].transform(
        lambda x: x.shift(1).rolling(long_w, min_periods=1).mean()
    )
    df["rolling_avg_points"] = driver_group["points"].transform(
        lambda x: x.shift(1).rolling(long_w, min_periods=1).mean()
    )

    df["position_delta"] = df["grid_position"] - df["finishing_position"]
    df["position_delta_trend"] = driver_group["position_delta"].transform(
        lambda x: x.shift(1).rolling(short_w, min_periods=1).mean()
    )

    # -- Circuit-specific history --
    if "circuit_short_name" in df.columns:
        circuit_driver = df.groupby(["driver_number", "circuit_short_name"])
        df["circuit_avg_finish"] = circuit_driver["finishing_position"].transform(
            lambda x: x.shift(1).expanding().mean()
        )
        df["circuit_race_count"] = circuit_driver["finishing_position"].transform(
            lambda x: x.shift(1).expanding().count()
        )
    else:
        df["circuit_avg_finish"] = 0.0
        df["circuit_race_count"] = 0

    # -- Team performance --
    if "team_name" in df.columns:
        team_group = df.groupby(
            ["team_name", "year"] if "year" in df.columns else ["team_name"]
        )
        df["team_season_avg_finish"] = team_group["finishing_position"].transform(
            lambda x: x.shift(1).expanding().mean()
        )
        df["team_points_per_race"] = team_group["points"].transform(
            lambda x: x.shift(1).expanding().mean()
        )
    else:
        df["team_season_avg_finish"] = 0.0
        df["team_points_per_race"] = 0.0

    # -- Reliability (DNF rates) --
    df["dnf_rate_season"] = driver_group["is_dnf"].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    if "circuit_short_name" in df.columns:
        circuit_driver2 = df.groupby(["driver_number", "circuit_short_name"])
        df["dnf_rate_circuit"] = circuit_driver2["is_dnf"].transform(
            lambda x: x.shift(1).expanding().mean()
        )
    else:
        df["dnf_rate_circuit"] = 0.0

    # -- Phase 1: New features --
    df = _build_qualifying_features(df)
    df = _build_weather_features(df)
    df = _build_stint_features(df)
    df = _build_circuit_type_feature(df)
    df = _build_teammate_delta(df)

    # -- Fill NaN features --
    df["grid_position"] = df["grid_position"].fillna(20.0)
    for col in FEATURE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # -- Save processed data --
    settings.ensure_dirs()
    out_path = settings.project_root / settings.data_processed_dir / "features.csv"

    keep_cols = (
        [
            "session_key",
            "driver_number",
            "full_name",
            "team_name",
            "circuit_short_name",
            "year",
            TARGET_COLUMN,
        ]
        + FEATURE_COLUMNS
    )
    keep_cols = [c for c in keep_cols if c in df.columns]
    result = df[keep_cols].copy()

    result.to_csv(out_path, index=False)
    logger.info(f"Saved {len(result)} feature rows to {out_path}")

    return result
