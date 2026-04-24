"""Model evaluation - metrics, baseline comparison, diagnostic plots, and MLflow logging."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from f1_predictor.config import settings
from f1_predictor.features.engineering import FEATURE_COLUMNS, TARGET_COLUMN

logger = logging.getLogger(__name__)


def evaluate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict[str, float]:
    """Compute regression metrics and compare to baseline.

    Baseline: grid_position == finishing_position.

    Returns:
        Dict of metric name -> value.
    """
    y_pred = model.predict(X)
    baseline_pred = X["grid_position"].values

    metrics = {
        "model_mae": mean_absolute_error(y, y_pred),
        "model_rmse": np.sqrt(mean_squared_error(y, y_pred)),
        "model_r2": r2_score(y, y_pred),
        "baseline_mae": mean_absolute_error(y, baseline_pred),
        "baseline_rmse": np.sqrt(mean_squared_error(y, baseline_pred)),
        "baseline_r2": r2_score(y, baseline_pred),
    }

    metrics["mae_improvement"] = (
        (metrics["baseline_mae"] - metrics["model_mae"]) / metrics["baseline_mae"] * 100
    )

    logger.info("=== Evaluation Results ===")
    logger.info(
        f"Model  MAE: {metrics['model_mae']:.3f}  |  "
        f"RMSE: {metrics['model_rmse']:.3f}  |  "
        f"R2: {metrics['model_r2']:.3f}"
    )
    logger.info(
        f"Baseline MAE: {metrics['baseline_mae']:.3f}  |  "
        f"RMSE: {metrics['baseline_rmse']:.3f}  |  "
        f"R2: {metrics['baseline_r2']:.3f}"
    )
    logger.info(f"MAE improvement over baseline: {metrics['mae_improvement']:.1f}%")

    return metrics


def plot_feature_importance(model, save_dir: Path | None = None) -> Path:
    """Plot and save XGBoost feature importance."""
    if save_dir is None:
        save_dir = settings.project_root / settings.plots_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    importances = model.feature_importances_
    features = FEATURE_COLUMNS
    indices = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(indices)), importances[indices], color="#E10600")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([features[i] for i in indices])
    ax.set_xlabel("Feature Importance (Gain)")
    ax.set_title("XGBoost Feature Importance - F1 Position Predictor")
    plt.tight_layout()

    path = save_dir / "feature_importance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Feature importance plot saved to {path}")
    return path


def plot_predictions_vs_actual(
    y_true: pd.Series,
    y_pred: np.ndarray,
    save_dir: Path | None = None,
) -> Path:
    """Scatter plot: predicted vs actual finishing positions."""
    if save_dir is None:
        save_dir = settings.project_root / settings.plots_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.4, s=20, c="#E10600")
    ax.plot([0, 22], [0, 22], "k--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Actual Finishing Position")
    ax.set_ylabel("Predicted Finishing Position")
    ax.set_title("Predicted vs Actual Finishing Position")
    ax.legend()
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 22)
    plt.tight_layout()

    path = save_dir / "predictions_vs_actual.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Predictions vs actual plot saved to {path}")
    return path


def plot_residuals(
    y_true: pd.Series,
    y_pred: np.ndarray,
    save_dir: Path | None = None,
) -> Path:
    """Residual distribution plot."""
    if save_dir is None:
        save_dir = settings.project_root / settings.plots_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    residuals = y_true.values - y_pred

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(residuals, bins=30, color="#E10600", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Residual (Actual - Predicted)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Residual Distribution (mean={np.mean(residuals):.2f}, std={np.std(residuals):.2f})"
    )
    plt.tight_layout()

    path = save_dir / "residuals.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Residuals plot saved to {path}")
    return path


def run_evaluation_pipeline(
    model_path: Path | None = None,
    features_path: Path | None = None,
    track: bool = True,
    run_id: str | None = None,
) -> dict[str, float]:
    """End-to-end evaluation: load model + data -> metrics -> plots -> MLflow.

    Args:
        model_path: Path to the joblib model file.
        features_path: Path to the features CSV.
        track: Whether to log results to MLflow.
        run_id: If provided, log to an existing MLflow run instead of creating new.

    Returns:
        Dict of evaluation metrics.
    """
    if model_path is None:
        model_path = settings.model_path
    if features_path is None:
        features_path = settings.project_root / settings.data_processed_dir / "features.csv"

    model = joblib.load(model_path)
    df = pd.read_csv(features_path)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    metrics = evaluate_model(model, X, y)

    y_pred = model.predict(X)
    importance_plot = plot_feature_importance(model)
    pred_vs_actual_plot = plot_predictions_vs_actual(y, y_pred)
    residuals_plot = plot_residuals(y, y_pred)

    # Log to MLflow
    if track:
        import mlflow

        from f1_predictor.tracking.mlflow_tracker import F1Tracker

        tracker = F1Tracker()

        if run_id:
            # Log to existing run (e.g., the training run)
            with mlflow.start_run(run_id=run_id):
                tracker.log_evaluation_metrics(metrics)
                tracker.log_artifact(importance_plot, artifact_path="plots")
                tracker.log_artifact(pred_vs_actual_plot, artifact_path="plots")
                tracker.log_artifact(residuals_plot, artifact_path="plots")
                logger.info(f"Evaluation metrics logged to existing run {run_id}")
        else:
            # Create a new evaluation run
            with tracker.start_run(
                run_name="evaluation",
                tags={"run_type": "evaluation"},
            ):
                tracker.log_evaluation_metrics(metrics)
                tracker.log_artifact(importance_plot, artifact_path="plots")
                tracker.log_artifact(pred_vs_actual_plot, artifact_path="plots")
                tracker.log_artifact(residuals_plot, artifact_path="plots")
                logger.info("Evaluation metrics logged to new MLflow run")

    return metrics
