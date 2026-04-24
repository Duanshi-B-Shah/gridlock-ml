"""Deploy the F1 predictor to SageMaker Serverless Endpoint.

Uses boto3 directly (no SageMaker Python SDK high-level classes) to avoid
version compatibility issues between sagemaker v2 and v3.

Usage:
    python infra/sagemaker/deploy.py --bucket my-f1-bucket
    python infra/sagemaker/deploy.py --bucket my-f1-bucket --delete

Prerequisites:
    pip install boto3
    AWS credentials configured (ada / aws configure)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
import tempfile
import time
from pathlib import Path

import boto3

# ── Configuration ──
MODEL_NAME = "f1-race-predictor"
ENDPOINT_NAME = "f1-predictor-serverless"
ROLE_NAME = "SageMakerF1PredictorRole"


def get_sklearn_image_uri(region: str) -> str:
    """Get the SKLearn container image URI for the given region."""
    # AWS-managed SKLearn container registry per region
    account_map = {
        "us-east-1": "683313688378",
        "us-east-2": "257758044811",
        "us-west-1": "746614075791",
        "us-west-2": "246618743249",
        "eu-west-1": "468650794304",
        "eu-west-2": "749696950732",
        "eu-central-1": "492215442770",
        "ap-northeast-1": "354813040037",
        "ap-southeast-1": "121021644041",
        "ap-southeast-2": "783357654285",
        "ap-south-1": "720646828776",
    }
    account = account_map.get(region, "683313688378")  # Default to us-east-1
    return f"{account}.dkr.ecr.{region}.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"


def get_or_create_role(iam_client) -> str:
    """Get or create the SageMaker execution role."""
    try:
        role = iam_client.get_role(RoleName=ROLE_NAME)
        return role["Role"]["Arn"]
    except iam_client.exceptions.NoSuchEntityException:
        pass

    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "sagemaker.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }

    role = iam_client.create_role(
        RoleName=ROLE_NAME,
        AssumeRolePolicyDocument=json.dumps(trust_policy),
        Description="SageMaker execution role for F1 Race Predictor",
    )

    iam_client.attach_role_policy(
        RoleName=ROLE_NAME,
        PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
    )
    iam_client.attach_role_policy(
        RoleName=ROLE_NAME,
        PolicyArn="arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess",
    )

    print("  Waiting for IAM role propagation...")
    time.sleep(10)
    return role["Role"]["Arn"]


def package_model(project_root: Path) -> Path:
    """Package model + inference code into model.tar.gz.

    Structure:
        xgb_f1_model.joblib
        quantile_models.joblib  (optional)
        code/
            inference.py
            requirements.txt
    """
    models_dir = project_root / "models"
    main_model = models_dir / "xgb_f1_model.joblib"
    quantile_model = models_dir / "quantile_models.joblib"

    if not main_model.exists():
        raise FileNotFoundError(
            f"Model not found at {main_model}. Run `python scripts/train_model.py` first."
        )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        shutil.copy2(main_model, tmp_path / "xgb_f1_model.joblib")
        if quantile_model.exists():
            shutil.copy2(quantile_model, tmp_path / "quantile_models.joblib")

        code_dir = tmp_path / "code"
        code_dir.mkdir()
        shutil.copy2(
            project_root / "infra" / "sagemaker" / "inference.py",
            code_dir / "inference.py",
        )

        # Only install xgboost + joblib — pandas/numpy/sklearn are pre-installed
        # Pinned versions to avoid numpy binary incompatibility
        (code_dir / "requirements.txt").write_text(
            "xgboost==2.0.3\njoblib==1.3.2\n"
        )

        tar_path = models_dir / "model.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            for root, dirs, files in os.walk(tmp_path):
                for file in files:
                    full_path = Path(root) / file
                    arcname = str(full_path.relative_to(tmp_path))
                    tar.add(full_path, arcname=arcname)

        print(f"  Packaged: {tar_path} ({tar_path.stat().st_size / 1024:.0f} KB)")
        return tar_path


def deploy(bucket: str, region: str, memory_mb: int = 2048, max_concurrency: int = 5) -> None:
    """Deploy model as SageMaker Serverless Endpoint using boto3 directly."""

    project_root = Path(__file__).resolve().parent.parent.parent
    sm = boto3.client("sagemaker", region_name=region)
    s3 = boto3.client("s3", region_name=region)
    iam = boto3.client("iam")

    ts = int(time.time())
    model_name = f"{MODEL_NAME}-{ts}"
    config_name = f"{MODEL_NAME}-config-{ts}"

    # 1. IAM Role
    print("1. Setting up IAM role...")
    role_arn = get_or_create_role(iam)
    print(f"  Role: {role_arn}")

    # 2. Package model
    print("\n2. Packaging model...")
    tar_path = package_model(project_root)

    # 3. Upload to S3
    print("\n3. Uploading to S3...")
    s3_key = "f1-predictor/models/model.tar.gz"
    s3.upload_file(str(tar_path), bucket, s3_key)
    s3_uri = f"s3://{bucket}/{s3_key}"
    print(f"  Uploaded: {s3_uri}")

    # 4. Get container image
    image_uri = get_sklearn_image_uri(region)
    print(f"\n4. Container: {image_uri}")

    # 5. Create Model
    print(f"\n5. Creating model: {model_name}")
    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": image_uri,
            "ModelDataUrl": s3_uri,
            "Environment": {
                "SAGEMAKER_PROGRAM": "inference.py",
                "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
            },
        },
        ExecutionRoleArn=role_arn,
    )

    # 6. Create Endpoint Config (Serverless)
    print(f"6. Creating endpoint config: {config_name}")
    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "ServerlessConfig": {
                    "MemorySizeInMB": memory_mb,
                    "MaxConcurrency": max_concurrency,
                },
            }
        ],
    )

    # 7. Create or Update Endpoint
    print(f"7. Deploying endpoint: {ENDPOINT_NAME}")
    try:
        sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
        sm.update_endpoint(
            EndpointName=ENDPOINT_NAME,
            EndpointConfigName=config_name,
        )
        print("  Updating existing endpoint...")
    except sm.exceptions.ClientError:
        sm.create_endpoint(
            EndpointName=ENDPOINT_NAME,
            EndpointConfigName=config_name,
        )
        print("  Creating new endpoint...")

    # 8. Wait
    print("8. Waiting for InService (2-5 minutes)...")
    waiter = sm.get_waiter("endpoint_in_service")
    waiter.wait(
        EndpointName=ENDPOINT_NAME,
        WaiterConfig={"Delay": 15, "MaxAttempts": 40},
    )

    print(f"\n✅ Endpoint deployed: {ENDPOINT_NAME}")
    print(f"   Region: {region}")
    print(f"\n   Test:")
    print(f"   python infra/sagemaker/test_endpoint.py --region {region}")


def delete_endpoint(region: str) -> None:
    """Tear down the endpoint and config."""
    sm = boto3.client("sagemaker", region_name=region)
    try:
        desc = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
        config_name = desc["EndpointConfigName"]
        print(f"  Deleting endpoint: {ENDPOINT_NAME}")
        sm.delete_endpoint(EndpointName=ENDPOINT_NAME)
        print(f"  Deleting config: {config_name}")
        sm.delete_endpoint_config(EndpointConfigName=config_name)
        print("  ✅ Deleted")
    except sm.exceptions.ClientError as e:
        print(f"  Not found: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy F1 predictor to SageMaker")
    parser.add_argument("--bucket", required=True, help="S3 bucket for model artifacts")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--memory", type=int, default=2048, help="Serverless memory in MB")
    parser.add_argument("--delete", action="store_true", help="Delete the endpoint")
    args = parser.parse_args()

    if args.delete:
        print("\n🗑️  Tearing down...")
        delete_endpoint(args.region)
        return

    print("\n🏎️  Deploying F1 Race Predictor to SageMaker Serverless\n")
    deploy(args.bucket, args.region, args.memory)
    print("\n🏁 Done!")


if __name__ == "__main__":
    main()
