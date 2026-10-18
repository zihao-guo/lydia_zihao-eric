**Live demo:** [Transaction labeling](https://zihao-guo.github.io/lydia_zihao-eric/) · [API docs](https://lydia-transaction-api-504999617745.europe-west1.run.app/docs)

# Transaction labeling — card-transaction category classifier

Each card transaction gets one of 12 minor categories and one of 8 major categories, using only fields available at payment time. The model is a sigmoid-calibrated linear SVM on character and word n-grams of the normalised title plus categorical and numeric features; the major category comes from a fixed taxonomy, a deterministic rule handles refunds, and a FastAPI service serves it. On the held-out final 20 % of the timeline (n = 1,601) the guarded accuracy (refund rule applied) is 0.955 (Wilson 95 % interval 0.944 to 0.964), macro-F1 0.941, major accuracy 0.964. That is next-period accuracy (95.1 % of test rows have a title seen in training); on the 78 unseen titles it is 0.795 (0.692 to 0.870).

Supporting material: [`notebooks/eda.ipynb`](notebooks/eda.ipynb), `models/metrics.json`, `models/model_comparison.csv`, `models/evaluation.json`, `models/model_card.json`.

## Repository layout

```
.
├── README.md                    this file
├── requirements.txt             pinned dependencies
├── data/
│   └── transactions.csv         raw data (8,121 rows)
├── notebooks/
│   └── eda.ipynb                executed exploratory analysis, sections 1-9
├── src/
│   ├── preprocessing.py         taxonomy, cleaning, split, features
│   ├── models.py                model zoo and hyper-parameter grids
│   ├── train.py                 training, comparison, artifact writing
│   ├── evaluate.py              evaluation, error buckets, provenance
│   ├── predict.py               Predictor, refund rule, command line
│   ├── train.sh                 install, train, evaluate
│   ├── api/
│   │   ├── main.py              FastAPI app and routes
│   │   ├── schemas.py           request and response models
│   │   └── index.html           web form served at /
│   ├── deploy/                  containers, hosting scripts and deployment guide
│   └── tests/                   pytest suite
└── models/
    ├── model.joblib             trained pipeline plus metadata
    ├── metrics.json             all candidates, legend
    ├── model_card.json          intended use, metrics, limitations
    ├── model_comparison.csv     the comparison as a table
    └── evaluation.json          test-period deep dive
```

## Approach and key design choices

- **Predict minor, derive major.** Minor determines major in every row; a second model would add inconsistent pairs.
- **Refund rule.** All 123 refund rows have `status == "refunded"` and `amount_value > 0`, and no other row does. Either condition fires it, overriding the probabilities.
- **Text.** Lower-cased, accent-stripped titles; char_wb 2-5-gram and word 1-2-gram TF-IDF (sublinear, `min_df=2`); char n-grams absorb spelling drift (McDonald's has 14 normalised variants).
- **Features.** Kept: entry method, digitalization, recurrence, status, `log1p(|amount|)`, sign, fees, time and round-amount flags. Dropped: `card_name` (mutual information 0.0286 vs a 0.0261 permutation baseline), identifiers, constants.
- **Deduplication before splitting:** 120 exact repeats removed, 8,001 rows remain.
- **Validation.** Temporal split: the first 80 % by `created_at` trains (6,400 rows), the last 1,601 test, from 2026-05-24. Selection uses 5-fold stratified CV on the training period; a 5-fold StratifiedGroupKFold by normalised title gives the cold-start view.
- **Selection rule:** CV accuracy, then CV macro-F1, then CV log-loss, then model name.
- **Model zoo.** Majority class, title lookup, logistic regression, calibrated linear SVC, text-only SVC ablation.
- **One artifact, shared preprocessing.** The artifact holds the whole pipeline and every stage imports `src/preprocessing.py`, so serving cannot drift from training.

The exploratory analysis behind these choices (nulls, duplicates, class balance, title spelling variants, label agreement per title, feature signal, amounts, time, identifiers) is in `notebooks/eda.ipynb`, sections 1–9.

## Result analysis

| model | cv_accuracy | coldstart_accuracy | test_accuracy (raw) | test_accuracy_guarded | test_macro_f1_guarded | seen_accuracy | unseen_accuracy |
|---|---|---|---|---|---|---|---|
| majority | 0.2892 | 0.3038 | 0.3061 | 0.3229 | 0.1229 | 0.3283 | 0.2179 |
| merchant_lookup | 0.8884 | 0.3038 | 0.9094 | 0.9263 | 0.8961 | 0.9626 | 0.2179 |
| linear_svc_text (ablation) | 0.9189 | 0.7284 | 0.9300 | 0.9469 | 0.9320 | 0.9573 | 0.7436 |
| logreg | 0.9364 | 0.7912 | 0.9525 | 0.9525 | 0.9396 | 0.9613 | 0.7821 |
| **linear_svc (selected)** | 0.9403 | 0.7989 | 0.9538 | 0.9550 | 0.9413 | 0.9632 | 0.7949 |

- `cv_accuracy`: 5-fold stratified CV on the training period, model alone; the selection criterion.
- `coldstart_accuracy`: title-grouped CV on the training period (no normalised title shared between folds), refund rule applied.
- `test_accuracy (raw)`: model alone on the held-out last 20 % of the timeline (1,601 rows).
- `test_accuracy_guarded`: same rows with the refund rule applied after prediction; the served behaviour and headline.
- `test_macro_f1_guarded`: unweighted mean F1 over the 12 classes, guarded.
- `seen_accuracy` / `unseen_accuracy`: guarded accuracy on test rows whose normalised title does / does not appear in the training period (1,523 / 78 rows).

The majority class is 29.3 % of rows.

- **Model vs title lookup.** Seen 0.9626 vs 0.9632, unseen 0.2179 vs 0.7949; overall +2.9 points guarded.
- **Structured features vs text-only ablation.** +2.1 CV points and +7.1 cold-start points; the raw test gap of 2.4 shrinks to 0.8 after the refund rule.
- **Logreg vs SVC** are within noise (2 test rows apart raw, 4 guarded); logreg has the better log-loss (CV 0.3008 vs 0.3078).
- **Errors.** 72 of 1,601 test rows are wrong (guarded), bucketed first match:

| Bucket | Definition | n |
|---|---|---|
| refund_rule | the rule is exact | 0 |
| cold_start | normalised title never seen in training | 16 |
| thin_support | title seen fewer than 5 times in training | 12 |
| title_majority | prediction is one of the title's most frequent training labels (5 or more training rows) | 44 |
| other | none of the above | 0 |

  title_majority is suspected, unverified label inconsistency. Examples (true -> predicted): `LE DIPLOMATE` supermarkets -> restaurants (cold_start); `SODEXO WEB` supermarkets -> restaurants (thin_support); `MC DONALD'S` transport_means -> restaurants (title_majority).
- **Confidence and coverage** (guarded, test period, from `evaluation.json`):

| Threshold on confidence | Covered | Accuracy on covered |
|---|---|---|
| 0.80 | 1,515 of 1,601 (94.6 %) | 0.9657 |
| 0.90 | 1,425 of 1,601 (89.0 %) | 0.9663 |
| 0.95 | 183 of 1,601 (11.4 %) | 0.9727 |

  The threshold is a product decision, to be set on validation data.
- **Caveats.** 1,601 test rows give about +-1 point, 78 unseen rows about +-9; cold-start CV measures unseen title strings, not merchants; hyper-parameters were chosen on the whole training period.

## How to run training

Python 3.11; run from the repository root.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest        # 204 tests, about 15 s
```

```bash
./src/train.sh              # one command: install pinned dependencies, train, evaluate
./src/train.sh --quick      # smoke run; no evaluation
```

Training writes `model.joblib`, `metrics.json`, `model_card.json` and `model_comparison.csv` to `--out`. `--quick` without `--out` goes to `/tmp/transaction-labeling-quick`, never over `models/`. Alternatives:

```bash
python -m src.train                  # all five models, full grid search
python -m src.train --quick --out /tmp/lydia-quick   # smoke test, kept out of models/
python -m src.train --models logreg,linear_svc --out /tmp/run1
```

| Flag | Meaning |
|---|---|
| `--models` | comma-separated subset of `majority, merchant_lookup, logreg, linear_svc, linear_svc_text` (default: all) |
| `--quick` | stratified subsample (about 1,200 train / 300 test rows), 2-fold CV, no grid search; same temporal cut |
| `--out` | output directory (default `models/`) |
| `--data` | input CSV (default `data/transactions.csv`) |

```bash
python -m src.evaluate               # re-checks the committed artifact on the test period
python -m src.predict --title "NETFLIX.COM" --amount -13.49
```

`src.evaluate` checks provenance (MD5, split date, row counts) and reproduces the stored accuracy; `src.predict` is the one-line command-line predictor.

## How to run the prediction service

Same environment as training:

```bash
uvicorn src.api.main:app --port 8000
LYDIA_MODEL_PATH=/path/to/model.joblib uvicorn src.api.main:app --port 8000   # another artifact
```

Web form at `http://localhost:8000/`, OpenAPI at `/docs`.

| Route | Purpose |
|---|---|
| `GET /health` | readiness: `{"status":"ok","model_loaded":true,"model_version":...}`; **503** when no model is loaded |
| `GET /v1/model` | artifact metadata and metrics |
| `POST /v1/predict` | one transaction -> one prediction |
| `POST /v1/predict/batch` | JSON list of transactions (at most 1,000) -> list of predictions |

Request body (`POST /v1/predict`; the batch route takes a list of the same objects):

| Field | Type | Required | Default | Note |
|---|---|---|---|---|
| `amount_value` | number | yes | | signed (negative = debit, positive = credit/refund), finite, absolute value at most 1e9 |
| `title` | string | one of the two | | merchant text; `title` or `merchant_name` must be non-blank |
| `merchant_name` | string | one of the two | | used when `title` is missing |
| `created_at` | string | no | | ISO 8601 only (date-only means midnight UTC); missing -> hour and weekday -1, weekend flag 0 |
| `status` | string | no | `completed` | `refunded`, or any positive `amount_value`, triggers the refund rule (prediction `refund`, confidence 1.0) |
| `merchant_pan_entry_method`, `digitalization_type`, `pfm_recurring_frequency` | string | no | | optional strings (e.g. `chip_contactless`, `apple_pay`, `monthly`); unknown values are ignored by the encoder |
| `fees` | integer | no | 0 | cents, 0 to 100,000,000 |
| `amount_currency`, `card_name`, `transaction_id` | string | no | `EUR` / null / null | accepted but unused (`transaction_id` is echoed back) |
| any other field | any | no | | accepted and ignored (e.g. `ref_id`, `member_id`, `card_type`) |

Example request and response:

```bash
curl -s -X POST localhost:8000/v1/predict -H 'Content-Type: application/json' \
  -d '{"title": "MC DONALD'"'"'S", "amount_value": -12.5, "created_at": "2026-06-01T12:30:00Z",
       "merchant_pan_entry_method": "chip_contactless", "digitalization_type": "apple_pay"}'
```

```json
{"transaction_id":null,"minor_category":"restaurants","major_category":"restaurants_bars_cafes","confidence":0.9527931024916834,"top_k":[{"label":"restaurants","probability":0.9527931024916834},{"label":"transport_means","probability":0.016257078203837688},{"label":"supermarkets","probability":0.009271843479240193}],"model_version":"linear_svc-20261018-154858"}
```

Errors: invalid input gives **422** with a list-shaped `detail`, for example `{"detail":[{"type":"missing","loc":["body","amount_value"],"msg":"Field required"}]}`; bodies are never logged. A missing model gives **503** `{"detail":"model not loaded"}`; an unloadable model file fails startup.

- The model is loaded once at startup (single process, CPU only); `model.joblib` is a pickle, so load only artifacts you trust.
- The three returned probabilities are the top 3 of 12 and need not sum to 1; `major_category` is `null` for `refund`.

## Assumptions and trade-offs

- **Synthetic data.** Real noise and merchants will differ.
- **Availability.** `status` and recurrence are assumed known at transaction time.
- **Recurrence encoding.** Missing `pfm_recurring_frequency` is encoded as `none` (non-recurring).
- **Holdout inspected.** I viewed the full frame, test period included, during EDA; selection used training-period CV only.
- **Single window, UTC.** 2026-03-13 to 2026-06-10: no seasonality or drift estimate; hour and weekday are UTC.

## How I imagined the rest

- A merchant-id cache of confirmed labels in front of the model, which stays as fallback.
- Relabel the high-volume mixed-label titles (`apple com bill`, `intermarche`, `carrefour city`).
- Per-class thresholds, human review of low-confidence rows, a reliability diagram first.
- Monitoring of unseen-title share, class mix and confidence drift.
- A cold-start model (char n-grams plus a small multilingual embedding), a model registry, and CI running the tests and a `--quick` training run.

## Reproducibility

- Seed 42 everywhere; scores and predictions are deterministic for the pinned versions, timings, timestamps and `model_version` are not.
- The artifact is committed, so the API runs without retraining.
- `python -m src.evaluate` re-checks it against the data (MD5, split date, row counts) and reproduces the stored accuracy (0.9550 guarded).
