"""CLI script: Train the XGBoost model with MLflow tracking."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from f1_predictor.training.train import run_training_pipeline
from f1_predictor.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the F1 position prediction model")
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Skip hyperparameter search (faster, uses defaults)",
    )
    parser.add_argument(
        "--features-path",
        type=str,
        default=None,
        help="Path to features CSV (default: data/processed/features.csv)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="MLflow experiment name (default: f1-race-predictor)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="MLflow run name (auto-generated if not set)",
    )
    parser.add_argument(
        "--no-track",
        action="store_true",
        help="Disable MLflow tracking",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="Register model in MLflow Model Registry",
    )
    args = parser.parse_args()

    setup_logging()

    features_path = Path(args.features_path) if args.features_path else None

    print("\n🏋️  Training model...")
    if not args.no_track:
        print("📊 MLflow tracking enabled")

    model, model_path = run_training_pipeline(
        features_path=features_path,
        tune=not args.no_tune,
        track=not args.no_track,
        experiment_name=args.experiment,
        run_name=args.run_name,
        register_model=args.register,
    )
    print(f"✓ Model saved to {model_path}")

    if not args.no_track:
        print("✓ Run logged to MLflow — view with: mlflow ui")


if __name__ == "__main__":
    main()
