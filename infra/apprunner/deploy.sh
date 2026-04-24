#!/bin/bash
# Deploy the F1 Race Predictor Streamlit app to AWS App Runner.
#
# Prerequisites:
#   - AWS CLI v2 configured (aws configure)
#   - Docker installed and running
#   - ECR repository created (or this script creates it)
#
# Usage:
#   ./infra/apprunner/deploy.sh                    # Deploy (creates everything)
#   ./infra/apprunner/deploy.sh --delete            # Tear down
#
# This script:
#   1. Builds the Docker image
#   2. Pushes to ECR
#   3. Creates/updates an App Runner service

set -euo pipefail

# ── Configuration ──
SERVICE_NAME="f1-race-predictor"
ECR_REPO_NAME="f1-race-predictor"
REGION="${AWS_DEFAULT_REGION:-us-east-1}"
PORT=8501

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${GREEN}[F1]${NC} $1"; }
warn() { echo -e "${YELLOW}[F1]${NC} $1"; }
error() { echo -e "${RED}[F1]${NC} $1"; exit 1; }

# ── Parse args ──
DELETE=false
if [[ "${1:-}" == "--delete" ]]; then
    DELETE=true
fi

# ── Get AWS account ID ──
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>/dev/null) || error "AWS CLI not configured. Run 'aws configure' first."
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO_NAME}"

# ── Delete mode ──
if $DELETE; then
    log "Tearing down App Runner service..."

    SERVICE_ARN=$(aws apprunner list-services --query "ServiceSummaryList[?ServiceName=='${SERVICE_NAME}'].ServiceArn" --output text 2>/dev/null || true)

    if [[ -n "$SERVICE_ARN" && "$SERVICE_ARN" != "None" ]]; then
        aws apprunner delete-service --service-arn "$SERVICE_ARN"
        log "Service deletion initiated: ${SERVICE_NAME}"
    else
        warn "Service not found: ${SERVICE_NAME}"
    fi

    log "Done. ECR repository retained (delete manually with: aws ecr delete-repository --repository-name ${ECR_REPO_NAME} --force)"
    exit 0
fi

# ── Step 1: Create ECR repository (if needed) ──
log "Step 1: Ensuring ECR repository exists..."
aws ecr describe-repositories --repository-names "$ECR_REPO_NAME" --region "$REGION" 2>/dev/null || \
    aws ecr create-repository --repository-name "$ECR_REPO_NAME" --region "$REGION" > /dev/null
log "  ECR: ${ECR_URI}"

# ── Step 2: Build Docker image ──
log "Step 2: Building Docker image..."
docker build -t "$ECR_REPO_NAME" .
log "  Image built: ${ECR_REPO_NAME}"

# ── Step 3: Push to ECR ──
log "Step 3: Pushing to ECR..."
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
docker tag "${ECR_REPO_NAME}:latest" "${ECR_URI}:latest"
docker push "${ECR_URI}:latest"
log "  Pushed: ${ECR_URI}:latest"

# ── Step 4: Create/Update App Runner service ──
log "Step 4: Deploying to App Runner..."

# Check if service exists
SERVICE_ARN=$(aws apprunner list-services --query "ServiceSummaryList[?ServiceName=='${SERVICE_NAME}'].ServiceArn" --output text 2>/dev/null || true)

if [[ -n "$SERVICE_ARN" && "$SERVICE_ARN" != "None" ]]; then
    log "  Updating existing service..."
    aws apprunner update-service \
        --service-arn "$SERVICE_ARN" \
        --source-configuration "{
            \"ImageRepository\": {
                \"ImageIdentifier\": \"${ECR_URI}:latest\",
                \"ImageRepositoryType\": \"ECR\",
                \"ImageConfiguration\": {
                    \"Port\": \"${PORT}\"
                }
            },
            \"AutoDeploymentsEnabled\": true
        }" > /dev/null
else
    log "  Creating new service..."

    # Create App Runner access role for ECR (if needed)
    ROLE_ARN=$(aws iam get-role --role-name AppRunnerECRAccessRole --query 'Role.Arn' --output text 2>/dev/null || true)
    if [[ -z "$ROLE_ARN" || "$ROLE_ARN" == "None" ]]; then
        log "  Creating ECR access role..."
        aws iam create-role \
            --role-name AppRunnerECRAccessRole \
            --assume-role-policy-document '{
                "Version": "2012-10-17",
                "Statement": [{
                    "Effect": "Allow",
                    "Principal": {"Service": "build.apprunner.amazonaws.com"},
                    "Action": "sts:AssumeRole"
                }]
            }' > /dev/null
        aws iam attach-role-policy \
            --role-name AppRunnerECRAccessRole \
            --policy-arn arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess
        sleep 10  # Wait for role propagation
        ROLE_ARN=$(aws iam get-role --role-name AppRunnerECRAccessRole --query 'Role.Arn' --output text)
    fi

    aws apprunner create-service \
        --service-name "$SERVICE_NAME" \
        --source-configuration "{
            \"ImageRepository\": {
                \"ImageIdentifier\": \"${ECR_URI}:latest\",
                \"ImageRepositoryType\": \"ECR\",
                \"ImageConfiguration\": {
                    \"Port\": \"${PORT}\"
                }
            },
            \"AutoDeploymentsEnabled\": true,
            \"AuthenticationConfiguration\": {
                \"AccessRoleArn\": \"${ROLE_ARN}\"
            }
        }" \
        --instance-configuration "{
            \"Cpu\": \"1024\",
            \"Memory\": \"2048\"
        }" \
        --health-check-configuration "{
            \"Protocol\": \"HTTP\",
            \"Path\": \"/_stcore/health\",
            \"Interval\": 10,
            \"Timeout\": 5,
            \"HealthyThreshold\": 1,
            \"UnhealthyThreshold\": 5
        }" > /dev/null
fi

# ── Step 5: Wait and get URL ──
log "Step 5: Waiting for deployment (this may take 2-5 minutes)..."
while true; do
    STATUS=$(aws apprunner list-services --query "ServiceSummaryList[?ServiceName=='${SERVICE_NAME}'].Status" --output text 2>/dev/null)
    if [[ "$STATUS" == "RUNNING" ]]; then
        break
    fi
    echo -n "."
    sleep 10
done
echo ""

SERVICE_URL=$(aws apprunner list-services --query "ServiceSummaryList[?ServiceName=='${SERVICE_NAME}'].ServiceUrl" --output text)

log ""
log "🏁 Deployment complete!"
log ""
log "  App URL:  https://${SERVICE_URL}"
log "  Service:  ${SERVICE_NAME}"
log "  Region:   ${REGION}"
log ""
log "  To tear down: ./infra/apprunner/deploy.sh --delete"
