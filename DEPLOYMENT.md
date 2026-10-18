# Live deployment

- Frontend: https://zihao-guo.github.io/lydia_zihao-eric/
- Google Cloud project: `lydia-512016` (project number `504999617745`).
- Region: `europe-west1`.
- Backend: Cloud Run service `lydia-transaction-api`, serving the committed `models/model.joblib` with FastAPI.
- Backend URL: https://lydia-transaction-api-504999617745.europe-west1.run.app
- Model registry: `lydia-transaction-classifier`, using the same serving image. The website calls Cloud Run; registering the model does not create a separate provisioned prediction endpoint.

## Frontend updates

The `Deploy frontend to GitHub Pages` workflow runs on changes to the frontend, builder, or workflow on `main`. It can also be run manually. The repository's `API_BASE_URL` Actions variable supplies the Cloud Run HTTPS origin. `scripts/build_pages.py` copies the frontend to `dist/index.html`, inserts the API origin, and adds `.nojekyll`. Only those static files are uploaded to Pages.

The FastAPI-served frontend keeps its empty API origin and works locally at `/`. Both single and CSV batch predictions, readiness checks, and the API documentation link use the configured origin in the Pages build.

## Backend updates

With Google Cloud CLI authenticated:

```bash
bash scripts/deploy_backend.sh
```

The build uses the dedicated `lydia-build@lydia-512016.iam.gserviceaccount.com` account with `roles/run.builder`, approved during setup. `.gcloudignore` uploads only runtime source, dependencies, and the fitted model; training data and notebooks are excluded. Runtime dependencies match the fitted artifact's versions.

Cloud Run exposes port 8000, uses 1 CPU and 512 MiB memory, scales to zero when idle, and has a maximum of 2 instances. Public access enables predictions from GitHub Pages. CORS permits `https://zihao-guo.github.io`. No Google Cloud credentials or API keys are included in the frontend.

After redeployment, register the new image as a new model version if model-registry tracking is needed. Read the deployed image digest with:

```bash
gcloud run revisions describe REVISION --project=lydia-512016 --region=europe-west1 --format='value(status.imageDigest)'
```

## Verification

Use the deployed backend origin for these requests:

```bash
curl "$API_BASE_URL/health"
curl "$API_BASE_URL/v1/predict" \
  -H 'Content-Type: application/json' \
  -H 'Origin: https://zihao-guo.github.io' \
  -d '{"title":"NETFLIX.COM","amount_value":-13.49}'
```

Readiness must return `model_loaded: true`; predictions must include the category, confidence, and model version. The browser's health indicator should show that same version.
