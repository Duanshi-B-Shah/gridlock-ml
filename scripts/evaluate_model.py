"""CLI script: Evaluate the trained model with MLflow logging."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from f1_predictor.training.evaluate import run_evaluation_pipeline
from f1_predictor.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the F1 position prediction model")
    parser.add_argument(
        "--no-track",
        action="store_true",
        help="Disable MLflow logging",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Log metrics to an existing MLflow run ID (e.g., the training run)",
    )
    args = parser.parse_args()

    setup_logging()

    print("\n📊 Evaluating model...")
    metrics = run_evaluation_pipeline(
        track=not args.no_track,
        run_id=args.run_id,
    )

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"  Model MAE:    {metrics['model_mae']:.3f}")
    print(f"  Model RMSE:   {metrics['model_rmse']:.3f}")
    print(f"  Model R²:     {metrics['model_r2']:.3f}")
    print(f"  Baseline MAE: {metrics['baseline_mae']:.3f}")
    print(f"  Improvement:  {metrics['mae_improvement']:.1f}%")
    print("=" * 50)
    print("\n✓ Plots saved to plots/")

    if not args.no_track:
        print("✓ Metrics logged to MLflow")


if __name__ == "__main__":
    main()
