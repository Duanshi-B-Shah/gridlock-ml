# Infrastructure — Deployment Guide

## Architecture

```
┌─────────────────────┐    ┌──────────────────────────┐
│   Streamlit App     │    │  SageMaker Serverless    │
│   (App Runner)      │    │  Endpoint                │
│                     │    │                          │
│  https://f1-pred... │    │  POST /invocations       │
│  Full 6-tab UI      │    │  JSON → P1-P20 + CI     │
│  ~$10-25/mo         │    │  ~$1-5/mo (pay/request)  │
│                     │    │                          │
│  Auto-scales,       │    │  Scales to zero,         │
│  auto-pauses        │    │  cold start ~5-10s       │
└─────────────────────┘    └──────────────────────────┘
         │                            │
         └────────── Both use ────────┘
                      │
              ┌───────────────┐
              │  S3 Bucket    │
              │  model.tar.gz │
              │  + data       │
              └───────────────┘
```

## Option A: App Runner (Streamlit UI)

Deploy the full Streamlit app to a public URL.

### Prerequisites
- AWS CLI v2 configured (`aws configure`)
- Docker installed and running

### Deploy
```bash
chmod +x infra/apprunner/deploy.sh
./infra/apprunner/deploy.sh
```

### What it does
1. Builds the Docker image from `Dockerfile`
2. Creates an ECR repository and pushes the image
3. Creates an App Runner service with health checks
4. Outputs the public URL: `https://xxxxx.region.apprunner.com`

### Tear down
```bash
./infra/apprunner/deploy.sh --delete
```

### Cost
- ~$10-25/mo with auto-pause enabled
- 1 vCPU, 2 GB RAM

---

## Option B: SageMaker Serverless Endpoint (API)

Deploy the model as a REST API endpoint.

### Prerequisites
- AWS CLI configured
- S3 bucket for model artifacts
- `pip install boto3 sagemaker`

### Deploy
```bash
python infra/sagemaker/deploy.py --bucket YOUR-S3-BUCKET
```

### What it does
1. Creates an IAM execution role (if needed)
2. Packages model + inference code into `model.tar.gz`
3. Uploads to S3
4. Creates a SageMaker Serverless Endpoint (scales to zero)

### Test
```bash
# Quick test
python infra/sagemaker/test_endpoint.py

# Or via AWS CLI
aws sagemaker-runtime invoke-endpoint \
    --endpoint-name f1-predictor-serverless \
    --content-type application/json \
    --body '{"grid_position": 3, "quali_position": 2}' \
    output.json
```

### API Format

**Request** (JSON):
```json
{
    "grid_position": 3,
    "quali_position": 2,
    "rolling_avg_finish_short": 4.5,
    "is_wet_race": 1
}
```
Missing features are filled with sensible defaults.

**Response**:
```json
{
    "predictions": [4],
    "confidence_intervals": {
        "lower_10": [2],
        "median": [4],
        "upper_90": [7]
    }
}
```

### Tear down
```bash
python infra/sagemaker/deploy.py --bucket YOUR-BUCKET --delete
```

### Cost
- ~$1-5/mo for demo traffic (pay per invocation)
- 2 GB memory, scales to zero when idle
- Cold start: ~5-10 seconds

---

## Both Together

For the full production setup:
```bash
# 1. Deploy the API
python infra/sagemaker/deploy.py --bucket my-f1-bucket

# 2. Deploy the UI
./infra/apprunner/deploy.sh
```

Your interviewer gets:
- A live URL to play with the full Streamlit app
- A serverless API endpoint for programmatic access
- Both scale to zero when idle (~$12-30/mo total)
