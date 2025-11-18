#!/usr/bin/env bash
set -euo pipefail

AWS_REGION="${AWS_REGION:-eu-west-1}"
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-<account-id>}"
ECR_REPO_NAME="${ECR_REPO_NAME:-iot-multitask-ids-api}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}:${IMAGE_TAG}"

aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

docker build -t "$ECR_REPO_NAME" -f deployment/Dockerfile.api .
docker tag "$ECR_REPO_NAME:latest" "$ECR_URI"
docker push "$ECR_URI"

echo "Pushed image: $ECR_URI"
