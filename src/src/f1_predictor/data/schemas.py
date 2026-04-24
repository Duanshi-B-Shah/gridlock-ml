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

    # Features
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
