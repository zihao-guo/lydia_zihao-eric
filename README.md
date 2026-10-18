**Live demo:** [Transaction labeling](https://zihao-guo.github.io/lydia_zihao-eric/) · [API docs](https://lydia-transaction-api-504999617745.europe-west1.run.app/docs)

# Transaction labeling

Classifies card transactions into 12 minor categories and derives 8 major categories from a fixed taxonomy. A sigmoid-calibrated linear SVM handles classification; a deterministic rule handles refunds. FastAPI serves the committed model without retraining.

## Try the live demo

![Live frontend: transaction inputs on the left; predicted category, confidence and top candidates on the right.](src/deploy/demo.png)

1. Click an example such as **Supermarket** to run a prediction.
2. For your own transaction, enter a title and amount, then click **Predict category**. Negative amounts are debits; positive amounts trigger the refund rule.
3. For batch predictions, select **CSV input**, upload or paste CSV containing `title` (or `merchant_name`) and `amount_value`, then click **Preview** and **Predict CSV**.

## Repository layout

```text
.
├── README.md
├── requirements.txt            pinned dependencies
├── data/transactions.csv       8,121 raw transactions
├── notebooks/eda.ipynb          executed exploratory analysis
├── src/
│   ├── preprocessing.py        cleaning, taxonomy, splits, features
│   ├── models.py               model definitions and parameter grids
│   ├── train.py                training and model selection
│   ├── evaluate.py             evaluation and provenance checks
│   ├── predict.py              shared predictor and CLI
│   ├── train.sh                install, train, evaluate
│   ├── api/                    FastAPI, schemas, frontend
│   ├── deploy/                 hosting scripts and deployment guide
│   └── tests/                  pytest suite
└── models/                     fitted model, metrics, model card
```

## Approach

- **Text:** normalised titles with character 2–5-gram and word 1–2-gram TF-IDF.
- **Structured features:** entry method, digitalization, recurrence, status, amount, fees and time. Identifiers and `card_name` are excluded.
- **Refund rule:** `status == "refunded"` or a positive amount overrides the model with `refund`.
- **Shared pipeline:** training and serving use the same preprocessing and fitted artifact.

After removing 120 duplicates, the first 6,400 transactions by time train the model; the final 1,601 form the test set, starting 2026-05-24. Selection uses five-fold stratified cross-validation on the training period. Title-grouped cross-validation provides an additional generalisation diagnostic.

## Results

Lydia's PFM task automatically classifies every transaction. If misclassifications have similar business costs, the proportion classified correctly is the most direct product metric. **Accuracy of the complete prediction flow is the primary business metric.**

All results below include preprocessing, model inference and the refund rule, evaluated on the same held-out test set.

- **Accuracy:** proportion of transactions assigned the correct minor category.
- **Macro-F1:** equal weight for each of the 12 classes; a constraint on performance across categories.
- **Unseen-title accuracy:** Accuracy on the 78 test transactions whose normalised title is absent from training; a generalisation constraint.

| Model | Accuracy (primary) | Macro-F1 | Unseen-title accuracy |
|---|---|---|---|
| Majority baseline | 32.29% | 0.1229 | 21.79% |
| Merchant lookup | 92.63% | 0.8961 | 21.79% |
| Text-only SVC | 94.69% | 0.9320 | 74.36% |
| Logistic regression | 95.25% | 0.9396 | 78.21% |
| **Calibrated linear SVC** | **95.50%** | **0.9413** | **79.49%** |

The selected model gets 1,529/1,601 transactions correct (Wilson 95% interval: 94.37–96.41%) and 62/78 unseen-title transactions correct (69.25–86.96%).

### Probability calibration

Evaluated separately on the same complete-flow probabilities:

| Diagnostic | Value | Definition |
|---|---|---|
| Top-label ECE | 0.0399 | Weighted confidence–Accuracy gap in 10 equal-width bins |
| Multiclass Brier score | 0.0857 | Mean summed squared probability error across 12 classes, unscaled |
| Log-loss | 0.2461 | Mean negative log probability of the true class |

Lower is better. ECE measures binned confidence agreement; Brier score and log-loss measure overall probability quality. Mean confidence is 91.68%, versus 95.50% Accuracy. Calibration and confidence thresholds should be fitted on validation data.

Detailed results: [metrics](models/metrics.json), [evaluation](models/evaluation.json), [model card](models/model_card.json). Feature analysis: [EDA notebook](notebooks/eda.ipynb).

## Run locally

Python 3.11; run from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest
uvicorn src.api.main:app --port 8000
```

Open `http://localhost:8000/` for the form or `/docs` for request schemas. Set `LYDIA_MODEL_PATH` to use another trusted artifact.

| Route | Purpose |
|---|---|
| `GET /health` | model readiness |
| `GET /v1/model` | model metadata |
| `POST /v1/predict` | one transaction |
| `POST /v1/predict/batch` | up to 1,000 transactions |

```bash
curl -s localhost:8000/v1/predict \
  -H 'Content-Type: application/json' \
  -d '{"title":"CARREFOUR MARKET","amount_value":-54.30}'
```

Responses contain the minor and major category, confidence, top three candidates and model version. Refunds have no major category. Invalid input returns **422**; a missing model returns **503**.

## Training and reproducibility

```bash
./src/train.sh                         # install, train, evaluate
./src/train.sh --quick                 # smoke run, no evaluation
python -m src.evaluate                # verify the committed artifact
python -m src.predict --title "NETFLIX.COM" --amount -13.49
```

Training accepts `--models`, `--data` and `--out`; see `python -m src.train --help`. Quick runs default to `/tmp/transaction-labeling-quick`. Evaluation checks the data hash, split date and row counts. Seed 42 and pinned dependencies make scores reproducible.

Hosting instructions: [deployment guide](src/deploy/README.md).

## Assumptions and limitations

- Misclassifications are assumed to have similar business costs; `status` and recurrence are assumed available at prediction time.
- Label noise is a working assumption pending manual review.
- Missing recurrence is encoded as `none`.
- Data is synthetic and covers one three-month window. Most test titles occur in training; unseen-title results cover only 78 transactions and measure title strings rather than merchant identities.
- The full dataset, including the test period, was viewed during EDA. Model selection used training-period cross-validation.
