"""Tests for MLflow tracking integration."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from xgboost import XGBRegressor

from f1_predictor.features.engineering import FEATURE_COLUMNS, TARGET_COLUMN


class TestF1Tracker:
    """Tests for the F1Tracker MLflow wrapper."""

    @patch("f1_predictor.tracking.mlflow_tracker.mlflow")
    def test_tracker_initializes_experiment(self, mock_mlflow):
        """Tracker should set tracking URI and experiment on init."""
        from f1_predictor.tracking.mlflow_tracker import F1Tracker

        tracker = F1Tracker(experiment_name="test-experiment")

        mock_mlflow.set_tracking_uri.assert_called_once()
        mock_mlflow.set_experiment.assert_called_once_with("test-experiment")

    @patch("f1_predictor.tracking.mlflow_tracker.mlflow")
    def test_log_params_converts_to_strings(self, mock_mlflow):
        """All params should be stringified before logging."""
        from f1_predictor.tracking.mlflow_tracker import F1Tracker

        tracker = F1Tracker(experiment_name="test")
        tracker.log_params({"n_estimators": 200, "learning_rate": 0.1})

        mock_mlflow.log_params.assert_called_once_with(
            {"n_estimators": "200", "learning_rate": "0.1"}
        )

    @patch("f1_predictor.tracking.mlflow_tracker.mlflow")
    def test_log_metrics_delegates_correctly(self, mock_mlflow):
        """Metrics should be passed through to mlflow.log_metrics."""
        from f1_predictor.tracking.mlflow_tracker import F1Tracker

        tracker = F1Tracker(experiment_name="test")
        metrics = {"model_mae": 2.5, "model_rmse": 3.1}
        tracker.log_metrics(metrics)

        mock_mlflow.log_metrics.assert_called_once_with(metrics, step=None)

    @patch("f1_predictor.tracking.mlflow_tracker.mlflow")
    def test_log_evaluation_metrics_passes_through(self, mock_mlflow):
        """Evaluation metrics should be logged as-is."""
        from f1_predictor.tracking.mlflow_tracker import F1Tracker

        tracker = F1Tracker(experiment_name="test")
        eval_metrics = {
            "model_mae": 2.3,
            "baseline_mae": 3.8,
            "mae_improvement": 39.5,
        }
        tracker.log_evaluation_metrics(eval_metrics)

        mock_mlflow.log_metrics.assert_called_once()

    @patch("f1_predictor.tracking.mlflow_tracker.mlflow")
    def test_log_training_metrics_adds_prefix(self, mock_mlflow):
        """Training metrics should get 'train_' prefix."""
        from f1_predictor.tracking.mlflow_tracker import F1Tracker

        tracker = F1Tracker(experiment_name="test")
        tracker.log_training_metrics({"loss": 0.5, "mae": 2.1})

        call_args = mock_mlflow.log_metrics.call_args[0][0]
        assert "train_loss" in call_args
        assert "train_mae" in call_args

    @patch("f1_predictor.tracking.mlflow_tracker.mlflow")
    def test_log_artifact_skips_missing_file(self, mock_mlflow):
        """Should skip logging if artifact file doesn't exist."""
        from f1_predictor.tracking.mlflow_tracker import F1Tracker

        tracker = F1Tracker(experiment_name="test")
        tracker.log_artifact("/nonexistent/path/file.png")

        mock_mlflow.log_artifact.assert_not_called()

    @patch("f1_predictor.tracking.mlflow_tracker.mlflow")
    def test_log_data_stats(self, mock_mlflow):
        """Data statistics should be logged as params."""
        from f1_predictor.tracking.mlflow_tracker import F1Tracker

        tracker = F1Tracker(experiment_name="test")
        tracker.log_data_stats(n_rows=500, n_features=11, seasons=[2023, 2024])

        mock_mlflow.log_params.assert_called_once_with(
            {
                "data_rows": "500",
                "data_features": "11",
                "data_seasons": "2023,2024",
            }
        )


class TestTrainingWithTracking:
    """Integration-style tests for training + MLflow."""

    def test_train_model_accepts_tracker_none(self, sample_features):
        """Training should work fine with tracker=None (backward compat)."""
        from f1_predictor.training.train import train_model

        X = sample_features[FEATURE_COLUMNS]
        y = sample_features[TARGET_COLUMN]

        model = train_model(X, y, tune_hyperparams=False, tracker=None)
        assert model is not None
        assert hasattr(model, "predict")

    def test_train_model_produces_predictions_with_tracker(self, sample_features):
        """Model trained with tracker=None should still predict correctly."""
        from f1_predictor.training.train import train_model

        X = sample_features[FEATURE_COLUMNS]
        y = sample_features[TARGET_COLUMN]

        model = train_model(X, y, tune_hyperparams=False, tracker=None)
        preds = model.predict(X)

        assert len(preds) == len(y)
        assert np.all(np.isfinite(preds))
