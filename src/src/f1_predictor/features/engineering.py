"""Feature engineering - transforms raw race results into model-ready features."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from f1_predictor.config import settings

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

    # -- Preprocessing: ensure required columns exist --
    # The fetcher now provides grid_position and finishing_position directly.
    # Fall back if raw data uses 'position' column (legacy format).
    if "finishing_position" not in df.columns and "position" in df.columns:
        df = df.rename(columns={"position": "finishing_position"})

    if "grid_position" not in df.columns:
        logger.warning("No 'grid_position' column found — defaulting to 20 (back of grid)")
        df["grid_position"] = 20.0

    df = _compute_dnf(df)
    df["grid_position"] = pd.to_numeric(df["grid_position"], errors="coerce").fillna(20.0)

    df["points"] = df["finishing_position"].apply(_assign_points)

    # Drop rows with no finishing position (incomplete data)
    df = df.dropna(subset=["finishing_position"])
    df["finishing_position"] = df["finishing_position"].astype(float)

    # Sort chronologically
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

    # Position delta trend (positions gained/lost)
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

    # -- Fill NaN features with sensible defaults --
    df["grid_position"] = df["grid_position"].fillna(20.0)  # Back of grid
    for col in FEATURE_COLUMNS:
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
    # Only keep columns that exist
    keep_cols = [c for c in keep_cols if c in df.columns]
    result = df[keep_cols].copy()

    result.to_csv(out_path, index=False)
    logger.info(f"Saved {len(result)} feature rows to {out_path}")

    return result
