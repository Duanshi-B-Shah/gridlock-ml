from __future__ import annotations

"""MLflow experiment tracking wrapper for the F1 predictor.

Provides a clean interface for:
- Experiment management (create/set active experiment)
- Parameter logging (hyperparams, feature config, data stats)
- Metric logging (training + evaluation metrics)
- Artifact logging (plots, feature CSVs, model files)
- Model registration (MLflow Model Registry)

Usage:
    tracker = F1Tracker(experiment_name="f1-race-predictor")
    with tracker.start_run(run_name="xgb-v1") as run:
        tracker.log_params({"n_estimators": 200, "max_depth": 5})
        tracker.log_training_metrics({"model_mae": 2.3})
        tracker.log_model(model, "xgb_model")
        tracker.log_artifact("plots/feature_importance.png")
"""

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

import mlflow
from mlflow.models import infer_signature

from f1_predictor.config import settings

logger = logging.getLogger(__name__)


class F1Tracker:
    """MLflow experiment tracker for the F1 Race Position Predictor."""

    def __init__(self, experiment_name: str | None = None) -> None:
        """Initialize the tracker.

        Args:
            experiment_name: MLflow experiment name. Defaults to config value.
        """
        self.experiment_name = experiment_name or settings.mlflow_experiment_name
        self._tracking_uri = settings.mlflow_tracking_uri
        self._active_run: mlflow.ActiveRun | None = None

        # Configure MLflow
        mlflow.set_tracking_uri(self._tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        logger.info(
            f"MLflow tracker initialized — experiment='{self.experiment_name}', "
            f"tracking_uri='{self._tracking_uri}'"
        )

    @contextmanager
    def start_run(
        self,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
        nested: bool = False,
    ) -> Generator[mlflow.ActiveRun, None, None]:
        """Start an MLflow run as a context manager.

        Args:
            run_name: Human-readable name for this run.
            tags: Optional tags to attach to the run.
            nested: Whether this is a nested run (e.g., for CV folds).

        Yields:
            The active MLflow run.
        """
        with mlflow.start_run(run_name=run_name, nested=nested) as run:
            self._active_run = run

            if tags:
                mlflow.set_tags(tags)

            # Always tag with project metadata
            mlflow.set_tags(
                {
                    "project": "f1-race-predictor",
                    "model_type": "xgboost",
                    "task": "regression",
                }
            )

            logger.info(f"MLflow run started — run_id={run.info.run_id}, name='{run_name}'")

            try:
                yield run
            finally:
                self._active_run = None
                logger.info(f"MLflow run ended — run_id={run.info.run_id}")

    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters to the active run.

        Args:
            params: Dict of parameter names to values.
        """
        # MLflow params must be strings — convert cleanly
        clean_params = {k: str(v) for k, v in params.items()}
        mlflow.log_params(clean_params)
        logger.info(f"Logged {len(params)} parameters")

    def log_data_stats(self, n_rows: int, n_features: int, seasons: list[int]) -> None:
        """Log dataset statistics as parameters.

        Args:
            n_rows: Number of training rows.
            n_features: Number of feature columns.
            seasons: List of seasons included in training data.
        """
        mlflow.log_params(
            {
                "data_rows": str(n_rows),
                "data_features": str(n_features),
                "data_seasons": ",".join(str(s) for s in seasons),
            }
        )

    def log_feature_config(self) -> None:
        """Log feature engineering configuration from settings."""
        mlflow.log_params(
            {
                "rolling_window_short": str(settings.rolling_window_short),
                "rolling_window_long": str(settings.rolling_window_long),
            }
        )

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics to the active run.

        Args:
            metrics: Dict of metric names to values.
            step: Optional step number (for tracking over time).
        """
        mlflow.log_metrics(metrics, step=step)
        logger.info(f"Logged {len(metrics)} metrics")

    def log_training_metrics(self, metrics: dict[str, float]) -> None:
        """Log training-specific metrics with a 'train_' prefix.

        Args:
            metrics: Dict of training metrics.
        """
        prefixed = {f"train_{k}" if not k.startswith("train_") else k: v for k, v in metrics.items()}
        self.log_metrics(prefixed)

    def log_evaluation_metrics(self, metrics: dict[str, float]) -> None:
        """Log evaluation metrics (model + baseline comparison).

        Args:
            metrics: Dict from evaluate_model() — includes model_mae, baseline_mae, etc.
        """
        self.log_metrics(metrics)

    def log_cv_results(self, cv_results: dict[str, float]) -> None:
        """Log cross-validation fold results.

        Args:
            cv_results: Dict with cv_mean_mae, cv_std_mae, etc.
        """
        self.log_metrics(cv_results)

    def log_artifact(self, local_path: str | Path, artifact_path: str | None = None) -> None:
        """Log a local file as an artifact.

        Args:
            local_path: Path to the file to log.
            artifact_path: Optional subdirectory within the artifact store.
        """
        path = Path(local_path)
        if not path.exists():
            logger.warning(f"Artifact not found, skipping: {path}")
            return

        mlflow.log_artifact(str(path), artifact_path=artifact_path)
        logger.info(f"Logged artifact: {path.name}")

    def log_artifacts_dir(self, local_dir: str | Path, artifact_path: str | None = None) -> None:
        """Log all files in a directory as artifacts.

        Args:
            local_dir: Path to the directory.
            artifact_path: Optional subdirectory within the artifact store.
        """
        path = Path(local_dir)
        if not path.exists() or not path.is_dir():
            logger.warning(f"Artifact directory not found, skipping: {path}")
            return

        mlflow.log_artifacts(str(path), artifact_path=artifact_path)
        logger.info(f"Logged artifacts directory: {path}")

    def log_model(
        self,
        model,
        artifact_path: str = "model",
        X_sample=None,
        register_name: str | None = None,
    ) -> None:
        """Log the trained model to MLflow.

        Args:
            model: The trained XGBoost model.
            artifact_path: Path within the run's artifact store.
            X_sample: Optional sample input for signature inference.
            register_name: If provided, register the model in MLflow Model Registry.
        """
        signature = None
        if X_sample is not None:
            predictions = model.predict(X_sample)
            signature = infer_signature(X_sample, predictions)

        mlflow.xgboost.log_model(
            model,
            artifact_path=artifact_path,
            signature=signature,
            registered_model_name=register_name,
        )

        if register_name:
            logger.info(f"Model logged and registered as '{register_name}'")
        else:
            logger.info(f"Model logged to artifact path '{artifact_path}'")

    @property
    def run_id(self) -> str | None:
        """Get the current active run ID."""
        if self._active_run:
            return self._active_run.info.run_id
        return None

    @staticmethod
    def get_best_run(
        experiment_name: str | None = None,
        metric: str = "model_mae",
        ascending: bool = True,
    ) -> dict[str, Any] | None:
        """Find the best run in an experiment by a given metric.

        Args:
            experiment_name: Experiment to search. Defaults to config.
            metric: Metric name to sort by.
            ascending: If True, lower is better (e.g., MAE).

        Returns:
            Dict with run_id, params, and metrics of the best run, or None.
        """
        exp_name = experiment_name or settings.mlflow_experiment_name
        order = "ASC" if ascending else "DESC"

        runs = mlflow.search_runs(
            experiment_names=[exp_name],
            order_by=[f"metrics.{metric} {order}"],
            max_results=1,
        )

        if runs.empty:
            return None

        best = runs.iloc[0]
        return {
            "run_id": best["run_id"],
            "metrics": {
                k.replace("metrics.", ""): v
                for k, v in best.items()
                if k.startswith("metrics.")
            },
            "params": {
                k.replace("params.", ""): v
                for k, v in best.items()
                if k.startswith("params.")
            },
        }

    @staticmethod
    def load_model_from_run(run_id: str, artifact_path: str = "model"):
        """Load a model from a specific MLflow run.

        Args:
            run_id: The MLflow run ID.
            artifact_path: Path to the model within the run's artifacts.

        Returns:
            The loaded XGBoost model.
        """
        model_uri = f"runs:/{run_id}/{artifact_path}"
        return mlflow.xgboost.load_model(model_uri)

    @staticmethod
    def load_production_model(model_name: str):
        """Load the latest production model from the registry.

        Args:
            model_name: Registered model name.

        Returns:
            The loaded model.
        """
        model_uri = f"models:/{model_name}@champion"
        return mlflow.xgboost.load_model(model_uri)
