"""OpenF1 API client - fetches session, position, and driver data."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests

from f1_predictor.config import settings

logger = logging.getLogger(__name__)

API = settings.openf1_base_url


def _get(endpoint: str, params: dict | None = None) -> list[dict]:
    """Make a GET request to the OpenF1 API with retry logic."""
    url = f"{API}/{endpoint}"
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.warning(f"API request failed (attempt {attempt + 1}/3): {e}")
            if attempt < 2:
                time.sleep(2**attempt)
    logger.error(f"Failed to fetch {endpoint} after 3 attempts")
    return []


def fetch_sessions(season: int) -> pd.DataFrame:
    """Fetch all race sessions for a given season."""
    logger.info(f"Fetching race sessions for {season}")
    data = _get("sessions", {"year": season, "session_type": "Race"})
    if not data:
        logger.warning(f"No sessions found for {season}")
        return pd.DataFrame()
    df = pd.DataFrame(data)
    logger.info(f"Found {len(df)} race sessions for {season}")
    return df


def fetch_positions(session_key: int) -> pd.DataFrame:
    """Fetch position data for a session and derive grid + finishing positions.

    The OpenF1 position endpoint returns time-series position data. We derive:
    - grid_position: first recorded position per driver (starting grid)
    - finishing_position: last recorded position per driver (final classification)
    """
    data = _get("position", {"session_key": session_key})
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)

    if "date" not in df.columns or "position" not in df.columns:
        logger.warning(f"Session {session_key}: missing 'date' or 'position' columns")
        return pd.DataFrame()

    df = df.sort_values("date")

    # Grid position = first recorded position per driver
    grid = (
        df.groupby("driver_number")
        .first()
        .reset_index()[["driver_number", "position"]]
        .rename(columns={"position": "grid_position"})
    )

    # Finishing position = last recorded position per driver
    finish = (
        df.groupby("driver_number")
        .last()
        .reset_index()[["driver_number", "position"]]
        .rename(columns={"position": "finishing_position"})
    )

    result = grid.merge(finish, on="driver_number", how="outer")
    return result


def fetch_drivers(session_key: int) -> pd.DataFrame:
    """Fetch driver info for a session."""
    data = _get("drivers", {"session_key": session_key})
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    # Deduplicate - keep first entry per driver
    if "driver_number" in df.columns:
        df = df.drop_duplicates(subset=["driver_number"], keep="first")
    return df


def fetch_and_save_season(season: int) -> Path:
    """Fetch all race data for a season and save to CSV.

    Returns:
        Path to the saved CSV file.
    """
    settings.ensure_dirs()
    raw_dir = settings.project_root / settings.data_raw_dir

    sessions_df = fetch_sessions(season)
    if sessions_df.empty:
        raise ValueError(f"No race sessions found for {season}")

    all_results = []

    for _, session in sessions_df.iterrows():
        sk = session["session_key"]
        circuit = session.get("circuit_short_name", "Unknown")
        country = session.get("country_name", "Unknown")
        logger.info(f"Processing {country} GP ({circuit}) - session {sk}")

        positions_df = fetch_positions(sk)
        drivers_df = fetch_drivers(sk)

        if positions_df.empty or drivers_df.empty:
            logger.warning(f"Skipping session {sk} - missing data")
            continue

        # Merge driver info with positions
        driver_cols = ["driver_number"]
        if "full_name" in drivers_df.columns:
            driver_cols.append("full_name")
        if "team_name" in drivers_df.columns:
            driver_cols.append("team_name")

        merged = positions_df.merge(
            drivers_df[driver_cols].drop_duplicates(),
            on="driver_number",
            how="left",
        )

        merged["session_key"] = sk
        merged["circuit_short_name"] = circuit
        merged["country_name"] = country
        merged["year"] = season
        merged["date_start"] = session.get("date_start", "")

        all_results.append(merged)
        time.sleep(0.5)  # Be nice to the API

    if not all_results:
        raise ValueError(f"No race results collected for {season}")

    combined = pd.concat(all_results, ignore_index=True)

    # Save raw data
    raw_path = raw_dir / f"race_results_{season}.csv"
    combined.to_csv(raw_path, index=False)
    logger.info(f"Saved {len(combined)} rows to {raw_path}")

    # Also save sessions metadata
    sessions_path = raw_dir / f"sessions_{season}.csv"
    sessions_df.to_csv(sessions_path, index=False)

    return raw_path
