"""Test the deployed SageMaker endpoint with sample predictions."""

from __future__ import annotations

import argparse
import json

import boto3


def test_single(endpoint_name: str, region: str) -> None:
    """Test with a single driver prediction."""
    runtime = boto3.client("sagemaker-runtime", region_name=region)

    payload = {
        "grid_position": 1,
        "quali_position": 1,
        "grid_quali_delta": 0,
        "rolling_avg_finish_short": 2.0,
        "rolling_avg_finish_long": 2.5,
        "rolling_avg_points": 22.0,
        "position_delta_trend": 1.0,
        "circuit_avg_finish": 2.0,
        "circuit_race_count": 5,
        "team_season_avg_finish": 3.0,
        "team_points_per_race": 20.0,
        "dnf_rate_season": 0.0,
        "dnf_rate_circuit": 0.0,
        "air_temperature": 25.0,
        "track_temperature": 40.0,
        "is_wet_race": 0,
        "humidity": 50.0,
        "wind_speed": 2.0,
        "n_pit_stops": 1,
        "is_street_circuit": 0,
        "teammate_delta_rolling": -2.0,
    }

    print(f"🏎️  Testing endpoint: {endpoint_name}")
    print(f"   Sending: pole-sitter with strong form\n")

    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload),
    )

    result = json.loads(response["Body"].read().decode("utf-8"))
    print(f"   Prediction: P{result['predictions'][0]}")

    if "confidence_intervals" in result:
        ci = result["confidence_intervals"]
        print(f"   Confidence:  P{ci['lower_10'][0]}–P{ci['upper_90'][0]} (80% CI)")

    print(f"\n   ✅ Endpoint working!")


def test_batch(endpoint_name: str, region: str) -> None:
    """Test with a batch of 3 drivers."""
    runtime = boto3.client("sagemaker-runtime", region_name=region)

    drivers = [
        {"grid_position": 1, "quali_position": 1, "rolling_avg_finish_short": 2.0},
        {"grid_position": 10, "quali_position": 9, "rolling_avg_finish_short": 10.0},
        {"grid_position": 20, "quali_position": 20, "rolling_avg_finish_short": 18.0},
    ]

    print(f"\n📋 Batch test: 3 drivers (P1, P10, P20 grid)\n")

    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(drivers),
    )

    result = json.loads(response["Body"].read().decode("utf-8"))
    for i, (driver, pred) in enumerate(zip(drivers, result["predictions"])):
        print(f"   Grid P{driver['grid_position']:2d} → Predicted P{pred}")

    print(f"\n   ✅ Batch prediction working!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the SageMaker F1 predictor endpoint")
    parser.add_argument("--endpoint", default="f1-predictor-serverless", help="Endpoint name")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    args = parser.parse_args()

    test_single(args.endpoint, args.region)
    test_batch(args.endpoint, args.region)


if __name__ == "__main__":
    main()
