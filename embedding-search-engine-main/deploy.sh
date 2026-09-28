#!/usr/bin/env bash
# deploy.sh — Build, tag, and push the Docker image to Amazon ECR.
#
# Prerequisites:
#   - Docker is running locally
#   - AWS CLI is installed and configured (or AWS env vars are set)
#   - The ECR repository already exists
#
# Usage:
#   export AWS_REGION=us-east-1
#   export AWS_ACCOUNT_ID=123456789012
#   export ECR_REPOSITORY=bedrock-search-engine
#   bash deploy.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Required environment variables
# ---------------------------------------------------------------------------
: "${AWS_REGION:?AWS_REGION is required (e.g. us-east-1)}"
: "${AWS_ACCOUNT_ID:?AWS_ACCOUNT_ID is required (your 12-digit account ID)}"
: "${ECR_REPOSITORY:?ECR_REPOSITORY is required (e.g. bedrock-search-engine)}"

IMAGE_TAG="${IMAGE_TAG:-latest}"
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}"

echo "===> Building Docker image..."
docker build -t "${ECR_REPOSITORY}:${IMAGE_TAG}" .

echo "===> Logging in to Amazon ECR..."
aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin \
    "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

echo "===> Tagging image..."
docker tag "${ECR_REPOSITORY}:${IMAGE_TAG}" "${ECR_URI}:${IMAGE_TAG}"

echo "===> Pushing image to ECR..."
docker push "${ECR_URI}:${IMAGE_TAG}"

echo ""
echo "=== Done ==="
echo "Image pushed to: ${ECR_URI}:${IMAGE_TAG}"
echo ""
echo "Next steps:"
echo "  1. Update your ECS task definition to use the new image URI above."
echo "  2. Deploy a new revision of your ECS service."
echo "  3. Verify the ALB health check at /health returns 200."
