"""CLI script: Compare MLflow experiment runs and find the best model."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import mlflow
import pandas as pd

from f1_predictor.config import settings
from f1_predictor.tracking.mlflow_tracker import F1Tracker
from f1_predictor.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare MLflow experiment runs")
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Experiment name (default: f1-race-predictor)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="model_mae",
        help="Metric to rank by (default: model_mae)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top runs to display (default: 10)",
    )
    args = parser.parse_args()

    setup_logging()

    experiment_name = args.experiment or settings.mlflow_experiment_name

    print(f"\n🔍 Comparing runs in experiment: '{experiment_name}'")
    print(f"   Ranked by: {args.metric} (lower is better)\n")

    # Search all runs
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        order_by=[f"metrics.{args.metric} ASC"],
        max_results=args.top_n,
    )

    if runs.empty:
        print("No runs found. Train a model first: python scripts/train_model.py")
        return

    # Display comparison table
    display_cols = ["run_id", "tags.mlflow.runName", "status"]

    # Add metric columns
    metric_cols = [c for c in runs.columns if c.startswith("metrics.")]
    param_cols = [c for c in runs.columns if c.startswith("params.")]

    # Select key metrics
    key_metrics = [
        "metrics.model_mae",
        "metrics.model_rmse",
        "metrics.model_r2",
        "metrics.baseline_mae",
        "metrics.mae_improvement",
        "metrics.cv_best_mae",
        "metrics.cv_mean_mae",
    ]
    key_params = [
        "params.n_estimators",
        "params.max_depth",
        "params.learning_rate",
        "params.tuning_method",
    ]

    show_cols = [c for c in display_cols + key_metrics + key_params if c in runs.columns]
    display = runs[show_cols].copy()

    # Clean column names
    display.columns = [c.replace("metrics.", "").replace("params.", "").replace("tags.mlflow.", "") for c in display.columns]

    print(display.to_string(index=False))

    # Highlight best
    print(f"\n🏆 Best run by {args.metric}:")
    best = F1Tracker.get_best_run(experiment_name, args.metric)
    if best:
        print(f"   Run ID:  {best['run_id']}")
        for k, v in best["metrics"].items():
            if isinstance(v, float):
                print(f"   {k}: {v:.4f}")


if __name__ == "__main__":
    main()
