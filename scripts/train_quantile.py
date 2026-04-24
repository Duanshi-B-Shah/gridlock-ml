"""CLI script: Train quantile regression models for confidence intervals."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from f1_predictor.explainability.confidence import QuantilePredictor
from f1_predictor.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train quantile regression models for prediction intervals"
    )
    parser.add_argument(
        "--lower", type=float, default=0.1, help="Lower quantile (default: 0.1 = 10th percentile)"
    )
    parser.add_argument(
        "--upper", type=float, default=0.9, help="Upper quantile (default: 0.9 = 90th percentile)"
    )
    args = parser.parse_args()

    setup_logging()

    print(f"\n📊 Training quantile models (q={args.lower}, q=0.5, q={args.upper})...")

    predictor = QuantilePredictor(lower_quantile=args.lower, upper_quantile=args.upper)
    predictor.train()
    path = predictor.save()

    print(f"✓ Quantile models saved to {path}")
    print("  → Use in Streamlit or via: QuantilePredictor.load().predict_single(features)")


if __name__ == "__main__":
    main()
