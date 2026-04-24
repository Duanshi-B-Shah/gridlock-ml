"""XGBoost model training with time-series cross-validation and MLflow tracking."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
from xgboost import XGBRegressor

from f1_predictor.config import settings
from f1_predictor.features.engineering import FEATURE_COLUMNS, TARGET_COLUMN

logger = logging.getLogger(__name__)


def load_training_data(path: Path | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Load feature-engineered data and split into X, y.

    Args:
        path: Path to features CSV. Defaults to processed dir.

    Returns:
        Tuple of (feature DataFrame, target Series).
    """
    if path is None:
        path = settings.project_root / settings.data_processed_dir / "features.csv"

    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows from {path}")

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    return X, y


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    tune_hyperparams: bool = True,
    tracker=None,
) -> XGBRegressor:
    """Train an XGBoost regressor with optional hyperparameter tuning.

    Args:
        X: Feature matrix.
        y: Target vector (finishing position).
        tune_hyperparams: Whether to run grid search CV.
        tracker: Optional F1Tracker instance for MLflow logging.

    Returns:
        Trained XGBRegressor.
    """
    tscv = TimeSeriesSplit(n_splits=5)

    if tune_hyperparams:
        logger.info("Running hyperparameter search with TimeSeriesSplit CV")
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1, 0.2],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        }

        base_model = XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )

        grid = GridSearchCV(
            base_model,
            param_grid,
            cv=tscv,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            verbose=1,
            refit=True,
        )
        grid.fit(X, y)

        logger.info(f"Best params: {grid.best_params_}")
        logger.info(f"Best CV MAE: {-grid.best_score_:.3f}")

        model = grid.best_estimator_

        # Log to MLflow
        if tracker:
            tracker.log_params(grid.best_params_)
            tracker.log_params({"tuning_method": "GridSearchCV", "cv_folds": "5"})
            tracker.log_cv_results(
                {
                    "cv_best_mae": -grid.best_score_,
                    "cv_candidates_tested": len(grid.cv_results_["mean_test_score"]),
                }
            )
    else:
        default_params = {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }

        logger.info("Training with default hyperparameters")
        model = XGBRegressor(
            **default_params,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X, y)

        # Log to MLflow
        if tracker:
            tracker.log_params(default_params)
            tracker.log_params({"tuning_method": "none"})

            # Run CV for metrics even without tuning
            cv_scores = cross_val_score(
                model, X, y, cv=tscv, scoring="neg_mean_absolute_error"
            )
            tracker.log_cv_results(
                {
                    "cv_mean_mae": -cv_scores.mean(),
                    "cv_std_mae": cv_scores.std(),
                }
            )

    # Log common params
    if tracker:
        tracker.log_params(
            {
                "objective": "reg:squarederror",
                "random_state": "42",
                "n_features": str(len(FEATURE_COLUMNS)),
            }
        )
        tracker.log_feature_config()
        tracker.log_data_stats(
            n_rows=len(X),
            n_features=len(FEATURE_COLUMNS),
            seasons=[],  # Populated by caller if available
        )

    return model


def save_model(model: XGBRegressor, path: Path | None = None) -> Path:
    """Serialize model to disk via joblib.

    Args:
        model: Trained XGBRegressor.
        path: Output path. Defaults to config model_path.

    Returns:
        Path where model was saved.
    """
    if path is None:
        path = settings.model_path

    settings.ensure_dirs()
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")
    return path


def run_training_pipeline(
    features_path: Path | None = None,
    tune: bool = True,
    track: bool = True,
    experiment_name: str | None = None,
    run_name: str | None = None,
    register_model: bool = False,
) -> tuple[XGBRegressor, Path]:
    """End-to-end training: load data -> train -> save -> track.

    Args:
        features_path: Path to features CSV.
        tune: Whether to run hyperparameter tuning.
        track: Whether to log to MLflow (default: True).
        experiment_name: MLflow experiment name override.
        run_name: MLflow run name (auto-generated if None).
        register_model: Whether to register the model in MLflow Model Registry.

    Returns:
        Tuple of (trained model, model file path).
    """
    X, y = load_training_data(features_path)

    if track:
        from f1_predictor.tracking.mlflow_tracker import F1Tracker

        tracker = F1Tracker(experiment_name=experiment_name)

        auto_run_name = run_name or f"xgb-{'tuned' if tune else 'default'}-{len(X)}rows"

        with tracker.start_run(run_name=auto_run_name) as run:
            model = train_model(X, y, tune_hyperparams=tune, tracker=tracker)
            model_path = save_model(model)

            # Log the model to MLflow
            registry_name = (
                settings.mlflow_model_registry_name if register_model else None
            )
            tracker.log_model(
                model,
                artifact_path="model",
                X_sample=X.head(5),
                register_name=registry_name,
            )

            # Log the joblib artifact too
            tracker.log_artifact(model_path, artifact_path="joblib")

            logger.info(f"Training complete — MLflow run_id={run.info.run_id}")
    else:
        model = train_model(X, y, tune_hyperparams=tune, tracker=None)
        model_path = save_model(model)

    return model, model_path
