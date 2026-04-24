"""Pydantic models for OpenF1 API responses and internal datasets."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SessionInfo(BaseModel):
    """OpenF1 session metadata."""

    session_key: int
    session_name: str = ""
    session_type: str = ""
    circuit_short_name: str = ""
    country_name: str = ""
    date_start: str = ""
    year: int = 0


class DriverResult(BaseModel):
    """A single driver's race result."""

    session_key: int
    driver_number: int
    full_name: str = ""
    team_name: str = ""
    grid_position: int | None = None
    finishing_position: int | None = None
    is_dnf: bool = False
    points: float = 0.0


class WeatherSummary(BaseModel):
    """Session-level weather summary."""

    air_temperature: float | None = None
    track_temperature: float | None = None
    humidity: float | None = None
    wind_speed: float | None = None
    is_wet_race: int = 0


class StintSummary(BaseModel):
    """Per-driver stint/pit strategy summary."""

    driver_number: int
    n_pit_stops: int = 1
    primary_compound: str = "MEDIUM"
    total_laps: int = 0


class RaceDataRow(BaseModel):
    """Feature-engineered row ready for training."""

    session_key: int
    driver_number: int
    full_name: str = ""
    team_name: str = ""
    circuit_short_name: str = ""
    year: int = 0

    # Target
    finishing_position: float

    # Original features
    grid_position: float
    rolling_avg_finish_short: float = Field(
        default=0.0, description="Rolling avg finish (short window)"
    )
    rolling_avg_finish_long: float = Field(
        default=0.0, description="Rolling avg finish (long window)"
    )
    rolling_avg_points: float = 0.0
    position_delta_trend: float = 0.0
    circuit_avg_finish: float = 0.0
    circuit_race_count: int = 0
    team_season_avg_finish: float = 0.0
    team_points_per_race: float = 0.0
    dnf_rate_season: float = 0.0
    dnf_rate_circuit: float = 0.0

    # Phase 1: Qualifying
    quali_position: float = 0.0
    grid_quali_delta: float = 0.0

    # Phase 1: Weather
    air_temperature: float = 0.0
    track_temperature: float = 0.0
    is_wet_race: int = 0
    humidity: float = 0.0
    wind_speed: float = 0.0

    # Phase 1: Strategy
    n_pit_stops: int = 1

    # Phase 1: Circuit type
    is_street_circuit: int = 0

    # Phase 1: Teammate comparison
    teammate_delta_rolling: float = 0.0
