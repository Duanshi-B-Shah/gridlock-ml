"""OpenF1 API client - fetches session, position, driver, weather, stint, and qualifying data."""

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
    for attempt in range(5):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                wait = min(10 * (attempt + 1), 60)
                logger.warning(f"Rate limited (429), waiting {wait}s before retry {attempt + 2}/5")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.warning(f"API request failed (attempt {attempt + 1}/5): {e}")
            if attempt < 4:
                time.sleep(5 * (attempt + 1))
    logger.error(f"Failed to fetch {endpoint} after 5 attempts")
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
    """Fetch position data for a session and derive grid + finishing positions."""
    data = _get("position", {"session_key": session_key})
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)

    if "date" not in df.columns or "position" not in df.columns:
        logger.warning(f"Session {session_key}: missing 'date' or 'position' columns")
        return pd.DataFrame()

    df = df.sort_values("date")

    grid = (
        df.groupby("driver_number")
        .first()
        .reset_index()[["driver_number", "position"]]
        .rename(columns={"position": "grid_position"})
    )

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
    if "driver_number" in df.columns:
        df = df.drop_duplicates(subset=["driver_number"], keep="first")
    return df


# ═══════════════════════════════════════════════════
# Phase 1: Weather, Stints, Qualifying fetchers
# ═══════════════════════════════════════════════════


def fetch_weather(session_key: int) -> pd.DataFrame:
    """Fetch weather data for a session and aggregate to session-level summary.

    Returns:
        Single-row DataFrame with: air_temperature, track_temperature,
        humidity, wind_speed, is_wet_race. Empty DataFrame if no data.
    """
    data = _get("weather", {"session_key": session_key})
    if not data:
        logger.warning(f"No weather data for session {session_key}")
        return pd.DataFrame()

    df = pd.DataFrame(data)

    summary = {
        "air_temperature": df["air_temperature"].mean() if "air_temperature" in df.columns else None,
        "track_temperature": df["track_temperature"].mean() if "track_temperature" in df.columns else None,
        "humidity": df["humidity"].mean() if "humidity" in df.columns else None,
        "wind_speed": df["wind_speed"].mean() if "wind_speed" in df.columns else None,
        "is_wet_race": int(df["rainfall"].max() > 0) if "rainfall" in df.columns else 0,
    }

    return pd.DataFrame([summary])


def fetch_stints(session_key: int) -> pd.DataFrame:
    """Fetch stint/pit strategy data for all drivers in a session.

    Returns:
        DataFrame with: driver_number, n_pit_stops, primary_compound, total_laps.
    """
    data = _get("stints", {"session_key": session_key})
    if not data:
        logger.warning(f"No stint data for session {session_key}")
        return pd.DataFrame()

    df = pd.DataFrame(data)

    if "driver_number" not in df.columns:
        return pd.DataFrame()

    results = []
    for dn, group in df.groupby("driver_number"):
        n_pit_stops = int(group["stint_number"].max()) - 1 if "stint_number" in group.columns else 0
        n_pit_stops = max(n_pit_stops, 0)

        if "compound" in group.columns and "lap_start" in group.columns and "lap_end" in group.columns:
            group = group.copy()
            group["stint_laps"] = pd.to_numeric(group["lap_end"], errors="coerce") - pd.to_numeric(group["lap_start"], errors="coerce") + 1
            valid = group.dropna(subset=["stint_laps"])
            if not valid.empty:
                primary_compound = valid.loc[valid["stint_laps"].idxmax(), "compound"]
                total_laps = int(valid["stint_laps"].sum())
            else:
                primary_compound = "UNKNOWN"
                total_laps = 0
        else:
            primary_compound = "UNKNOWN"
            total_laps = 0

        results.append({
            "driver_number": dn,
            "n_pit_stops": n_pit_stops,
            "primary_compound": primary_compound,
            "total_laps": total_laps,
        })

    return pd.DataFrame(results)


def fetch_qualifying_positions(meeting_key: int, season: int) -> pd.DataFrame:
    """Fetch qualifying positions for a race weekend.

    Finds the qualifying session (not sprint qualifying) for the given meeting,
    then derives each driver's final qualifying position.

    Returns:
        DataFrame with: driver_number, quali_position.
    """
    sessions_data = _get("sessions", {
        "meeting_key": meeting_key,
        "session_type": "Qualifying",
        "year": season,
    })

    if not sessions_data:
        logger.warning(f"No qualifying session found for meeting {meeting_key}")
        return pd.DataFrame()

    # Prefer "Qualifying" over "Sprint Qualifying"
    quali_session = None
    for s in sessions_data:
        if s.get("session_name") == "Qualifying":
            quali_session = s
            break
    if quali_session is None:
        quali_session = sessions_data[0]

    quali_sk = quali_session["session_key"]
    logger.info(f"Fetching qualifying positions from session {quali_sk} ({quali_session.get('session_name', '')})")

    pos_data = _get("position", {"session_key": quali_sk})
    if not pos_data:
        logger.warning(f"No position data for qualifying session {quali_sk}")
        return pd.DataFrame()

    pos_df = pd.DataFrame(pos_data)
    if "date" not in pos_df.columns or "position" not in pos_df.columns:
        return pd.DataFrame()

    pos_df = pos_df.sort_values("date")
    quali_positions = (
        pos_df.groupby("driver_number")
        .last()
        .reset_index()[["driver_number", "position"]]
        .rename(columns={"position": "quali_position"})
    )

    return quali_positions


# ═══════════════════════════════════════════════════
# Updated season fetcher
# ═══════════════════════════════════════════════════


def fetch_and_save_season(season: int) -> Path:
    """Fetch all race data for a season and save to CSV.

    Fetches: positions, drivers, weather, stints, and qualifying for each race.
    """
    settings.ensure_dirs()
    raw_dir = settings.project_root / settings.data_raw_dir

    sessions_df = fetch_sessions(season)
    if sessions_df.empty:
        raise ValueError(f"No race sessions found for {season}")

    all_results = []

    for _, session in sessions_df.iterrows():
        sk = session["session_key"]
        mk = session.get("meeting_key", None)
        circuit = session.get("circuit_short_name", "Unknown")
        country = session.get("country_name", "Unknown")
        logger.info(f"Processing {country} GP ({circuit}) - session {sk}")

        positions_df = fetch_positions(sk)
        drivers_df = fetch_drivers(sk)

        if positions_df.empty or drivers_df.empty:
            logger.warning(f"Skipping session {sk} - missing position/driver data")
            continue

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

        # --- Weather data (session-level) ---
        time.sleep(2.0)
        weather_df = fetch_weather(sk)
        if not weather_df.empty:
            for col in ["air_temperature", "track_temperature", "humidity", "wind_speed", "is_wet_race"]:
                if col in weather_df.columns:
                    merged[col] = weather_df[col].iloc[0]

        # --- Stint data (per-driver) ---
        time.sleep(2.0)
        stints_df = fetch_stints(sk)
        if not stints_df.empty:
            merged = merged.merge(stints_df, on="driver_number", how="left")

        # --- Qualifying positions (per-driver) ---
        if mk is not None:
            time.sleep(2.0)
            quali_df = fetch_qualifying_positions(mk, season)
            if not quali_df.empty:
                merged = merged.merge(quali_df, on="driver_number", how="left")

        all_results.append(merged)
        time.sleep(3.0)

    if not all_results:
        raise ValueError(f"No race results collected for {season}")

    combined = pd.concat(all_results, ignore_index=True)

    raw_path = raw_dir / f"race_results_{season}.csv"
    combined.to_csv(raw_path, index=False)
    logger.info(f"Saved {len(combined)} rows to {raw_path}")

    sessions_path = raw_dir / f"sessions_{season}.csv"
    sessions_df.to_csv(sessions_path, index=False)

    return raw_path
