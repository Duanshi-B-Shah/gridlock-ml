"""Configuration management via pydantic-settings."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    # OpenF1 API
    openf1_base_url: str = "https://api.openf1.org/v1"

    # Paths
    project_root: Path = Path(__file__).resolve().parent.parent.parent
    data_raw_dir: Path = Path("data/raw")
    data_processed_dir: Path = Path("data/processed")
    models_dir: Path = Path("models")
    plots_dir: Path = Path("plots")

    # Model
    model_filename: str = "xgb_f1_model.joblib"

    # MLflow
    mlflow_tracking_uri: str = "mlruns"
    mlflow_experiment_name: str = "f1-race-predictor"
    mlflow_model_registry_name: str = "f1-position-predictor"

    # Feature engineering
    rolling_window_short: int = 3
    rolling_window_long: int = 5

    # Logging
    log_level: str = "INFO"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @property
    def model_path(self) -> Path:
        return self.project_root / self.models_dir / self.model_filename

    def ensure_dirs(self) -> None:
        """Create all output directories if they don't exist."""
        for d in [self.data_raw_dir, self.data_processed_dir, self.models_dir, self.plots_dir]:
            (self.project_root / d).mkdir(parents=True, exist_ok=True)


settings = Settings()
