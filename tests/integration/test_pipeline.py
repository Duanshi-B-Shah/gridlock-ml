"""Integration test: end-to-end pipeline (fetch -> features -> train -> predict)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error

from f1_predictor.features.engineering import FEATURE_COLUMNS, TARGET_COLUMN, build_features
from f1_predictor.training.train import train_model


class TestPipeline:
    def test_features_to_training_roundtrip(self, sample_raw_data, tmp_path):
        """Raw data -> features -> training should produce a working model."""
        csv_path = tmp_path / "raw.csv"
        sample_raw_data.to_csv(csv_path, index=False)

        features_df = build_features([csv_path])

        assert len(features_df) > 0
        assert all(col in features_df.columns for col in FEATURE_COLUMNS)

        X = features_df[FEATURE_COLUMNS]
        y = features_df[TARGET_COLUMN]

        model = train_model(X, y, tune_hyperparams=False)

        preds = model.predict(X)
        assert len(preds) == len(y)
        assert np.all(np.isfinite(preds))

    def test_model_beats_random(self, sample_raw_data, tmp_path):
        """Trained model should have lower MAE than random predictions."""
        csv_path = tmp_path / "raw.csv"
        sample_raw_data.to_csv(csv_path, index=False)

        features_df = build_features([csv_path])
        X = features_df[FEATURE_COLUMNS]
        y = features_df[TARGET_COLUMN]

        model = train_model(X, y, tune_hyperparams=False)
        model_preds = model.predict(X)

        np.random.seed(99)
        random_preds = np.random.uniform(1, 20, len(y))

        model_mae = mean_absolute_error(y, model_preds)
        random_mae = mean_absolute_error(y, random_preds)

        assert model_mae < random_mae, (
            f"Model MAE ({model_mae:.2f}) should be lower than random ({random_mae:.2f})"
        )
