"""CLI script: Fetch F1 race data from OpenF1 API."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from f1_predictor.data.fetcher import fetch_and_save_season
from f1_predictor.features.engineering import build_features
from f1_predictor.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch F1 race data from OpenF1 API")
    parser.add_argument(
        "--season",
        type=int,
        nargs="+",
        default=[2024],
        help="Season year(s) to fetch (default: 2024)",
    )
    parser.add_argument(
        "--driver",
        type=str,
        nargs="*",
        default=None,
        help="Filter by driver name(s) after fetching (e.g. --driver Verstappen Hamilton)",
    )
    parser.add_argument(
        "--driver-number",
        type=int,
        nargs="*",
        default=None,
        help="Filter by driver number(s) (e.g. --driver-number 1 44)",
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Skip feature engineering after fetching",
    )
    args = parser.parse_args()

    setup_logging()

    raw_paths = []
    for season in args.season:
        print(f"\n🏎️  Fetching {season} season data...")
        path = fetch_and_save_season(season)
        raw_paths.append(path)
        print(f"✓ Saved to {path}")

    if not args.skip_features:
        print("\n⚙️  Building features...")
        df = build_features(raw_paths)

        # Apply driver filters
        filtered = df.copy()

        if args.driver:
            # Case-insensitive partial match on full_name
            if "full_name" in filtered.columns:
                pattern = "|".join(args.driver)
                filtered = filtered[
                    filtered["full_name"].str.contains(pattern, case=False, na=False)
                ]
                print(f"🔍 Filtered to drivers matching: {', '.join(args.driver)}")
            else:
                print("⚠️  No 'full_name' column — driver name filter skipped")

        if args.driver_number:
            if "driver_number" in filtered.columns:
                filtered = filtered[filtered["driver_number"].isin(args.driver_number)]
                print(f"🔍 Filtered to driver numbers: {args.driver_number}")
            else:
                print("⚠️  No 'driver_number' column — driver number filter skipped")

        if args.driver or args.driver_number:
            if filtered.empty:
                print("⚠️  No rows matched the driver filter!")
            else:
                print(f"\n📋 Filtered dataset: {len(filtered)} rows")
                if "full_name" in filtered.columns:
                    drivers = filtered["full_name"].unique()
                    for d in sorted(drivers):
                        count = len(filtered[filtered["full_name"] == d])
                        print(f"   {d}: {count} races")

        print(f"\n✓ Full feature dataset: {len(df)} rows, {len(df.columns)} columns")
        if args.driver or args.driver_number:
            print(f"✓ Filtered dataset: {len(filtered)} rows")


if __name__ == "__main__":
    main()
