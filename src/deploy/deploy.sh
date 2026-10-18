#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

deploy_context="$(mktemp -d)"
trap 'rm -rf "$deploy_context"' EXIT
mkdir -p "$deploy_context/src" "$deploy_context/models"
cp src/deploy/Dockerfile src/deploy/.dockerignore src/deploy/.gcloudignore \
  src/deploy/requirements-runtime.txt "$deploy_context/"
cp src/__init__.py src/preprocessing.py src/models.py src/evaluate.py src/predict.py \
  "$deploy_context/src/"
cp -R src/api "$deploy_context/src/api"
cp models/model.joblib models/model_card.json "$deploy_context/models/"

# Requires gcloud login and the project's lydia-build service account with roles/run.builder.
gcloud run deploy lydia-transaction-api \
  --source="$deploy_context" \
  --project=lydia-512016 \
  --region=europe-west1 \
  --build-service-account=projects/lydia-512016/serviceAccounts/lydia-build@lydia-512016.iam.gserviceaccount.com \
  --port=8000 \
  --allow-unauthenticated \
  --memory=512Mi --cpu=1 \
  --min=0 --max=2 --concurrency=20 --timeout=60 \
  --set-env-vars=LYDIA_CORS_ORIGINS=https://zihao-guo.github.io \
  --quiet

gcloud run services describe lydia-transaction-api \
  --project=lydia-512016 --region=europe-west1 --format='value(status.url)'
