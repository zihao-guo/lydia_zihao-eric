# Transaction labeling — card-transaction category classifier

Each card transaction gets one of 12 minor categories and one of 8 major categories, using only the fields available when the payment is made. The model is a sigmoid-calibrated linear SVM on character and word n-grams of the normalised title plus categorical and numeric features. The major category comes from a fixed minor-to-major taxonomy, a deterministic rule handles refunds, and a FastAPI service with a small web form serves it. On the held-out final 20 % of the timeline (n = 1,601) the guarded accuracy (refund rule applied) is 0.955 (Wilson 95 % interval 0.944 to 0.964), macro-F1 0.941 and major accuracy 0.964. That is next-period accuracy under the current merchant mix: 95.1 % of test rows have a title seen in training. On the 78 unseen titles accuracy is 0.795 (0.692 to 0.870), and the title-grouped cold-start CV on the training period gives 0.799.

Supporting material: [`notebooks/eda.ipynb`](notebooks/eda.ipynb) (EDA), `models/metrics.json` and `models/model_comparison.csv` (model comparison), `models/evaluation.json` (test-period error analysis), `models/model_card.json`.

## Repository layout

```
README.md                      this file
requirements.txt               pinned dependencies
data/transactions.csv          raw data (8,121 rows)
notebooks/eda.ipynb            executed exploratory analysis, sections 1-9
src/preprocessing.py           taxonomy, normalisation, cleaning, split, features
src/models.py                  model zoo and hyper-parameter grids
src/train.py                   training, comparison, artifact writing
src/evaluate.py                evaluation, error buckets, provenance checks
src/predict.py                 Predictor class, refund rule, command-line prediction
src/train.sh                   one command: install pinned dependencies, train, evaluate
src/tests/                     pytest suite (preprocessing, training, evaluation, predictor, API and schemas); warnings are errors
src/api/main.py                FastAPI app and routes
src/api/schemas.py             request and response models with validation
src/api/index.html             web form served at /
models/model.joblib            the trained pipeline plus metadata
models/metrics.json            all candidates, legend, notes
models/model_card.json         description, intended use, metrics, limitations
models/model_comparison.csv    the comparison as a table
models/evaluation.json         test-period deep dive
```

## Setup

Python 3.11. Run every command from the repository root.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest        # 204 tests, about 15 s
```

## How to run training

```bash
./src/train.sh              # one command: install pinned dependencies, train, evaluate
./src/train.sh --quick      # smoke run; skips the evaluation step
```

`src/train.sh` runs `pip install -r requirements.txt`, then `python -m src.train "$@"`, then `python -m src.evaluate` unless `--quick` is given. With `--quick` and no `--out`, it writes to `/tmp/transaction-labeling-quick`, so the shipped artifact in `models/` is never overwritten. The individual training commands are alternatives, not a sequence:

```bash
python -m src.train                  # all five models, full search
python -m src.train --quick --out /tmp/lydia-quick   # smoke test, kept out of models/
python -m src.train --models logreg,linear_svc --out /tmp/run1
```

| Flag | Meaning |
|---|---|
| `--models` | comma-separated subset of `majority, merchant_lookup, logreg, linear_svc, linear_svc_text` (default: all) |
| `--quick` | stratified subsample (about 1,200 train / 300 test rows), 2-fold CV, no grid search; same temporal cut |
| `--out` | output directory (default `models/`) |
| `--data` | input CSV (default `data/transactions.csv`) |

Search plus refit took 196 s in total (about 92 to 95 s for each grid-searched linear model, from `model_comparison.csv`); the whole run, with the cold-start CV and the evaluation, takes about 4 minutes. It writes `model.joblib`, `metrics.json`, `model_card.json` and `model_comparison.csv` to `--out`. A `--quick` artifact is flagged in its metadata, and `src.evaluate` refuses it unless `--force` is given.

```bash
python -m src.evaluate               # re-checks the committed artifact on the test period
python -m src.predict --title "NETFLIX.COM" --amount -13.49
```

`src.evaluate` first checks provenance (MD5 of the data file, split date, row counts against the artifact metadata) and stops on a mismatch. It then prints raw vs guarded accuracy with Wilson intervals, seen vs unseen titles, the error-bucket table and the confidence/coverage table, and writes `evaluation.json` (flags `--model`, `--data`, `--out`, `--force`). `src.predict` also takes `--json` (an object or a list), `--merchant-name`, `--status`, `--created-at`, `--entry-method`, `--digitalization`, `--recurring`, `--fees` and `--model`.

## How to run the prediction service

```bash
uvicorn src.api.main:app --port 8000
LYDIA_MODEL_PATH=/path/to/model.joblib uvicorn src.api.main:app --port 8000   # another artifact
```

The web form is at `http://localhost:8000/`, the OpenAPI documentation at `/docs`.

| Route | Purpose |
|---|---|
| `GET /health` | readiness: `{"status":"ok","model_loaded":true,"model_version":...}`; **503** with `"status":"unavailable"` when no model is loaded |
| `GET /v1/model` | artifact metadata (version, training rows and split, CV/cold-start/test metrics, features, selection rule) |
| `POST /v1/predict` | one transaction -> one prediction |
| `POST /v1/predict/batch` | JSON list of transactions (at most 1,000) -> list of predictions |

Request body (`POST /v1/predict`; the batch route takes a list of the same objects):

| Field | Type | Required | Default | Note |
|---|---|---|---|---|
| `amount_value` | number | yes | | signed (negative = debit, positive = credit/refund), finite, absolute value at most 1e9; numeric strings accepted, booleans rejected |
| `title` | string | one of the two | | merchant text; `title` or `merchant_name` must be non-blank |
| `merchant_name` | string | one of the two | | used when `title` is missing; non-string text counts as missing |
| `created_at` | string | no | | ISO 8601 only (date-only means midnight UTC); numeric timestamps rejected; missing -> hour and weekday are set to -1 and the weekend flag to 0 |
| `status` | string | no | `completed` | `refunded`, or any positive `amount_value`, triggers the refund rule (prediction `refund`, confidence 1.0) |
| `merchant_pan_entry_method` | string | no | | e.g. `chip`, `chip_contactless` |
| `digitalization_type` | string | no | | e.g. `apple_pay`, `none` |
| `pfm_recurring_frequency` | string | no | | e.g. `monthly` |
| `fees` | integer | no | 0 | cents, 0 to 100,000,000 |
| `amount_currency` | string | no | `EUR` | accepted but unused |
| `card_name` | string | no | null | accepted but unused |
| `transaction_id` | string | no | null | echoed back as a string |
| any other field | any | no | | accepted and ignored (e.g. `ref_id`, `member_id`, `member_name`, `merchant_id`, `card_type`) |

Example request and response:

```bash
curl -s -X POST localhost:8000/v1/predict -H 'Content-Type: application/json' \
  -d '{"title": "MC DONALD'"'"'S", "amount_value": -12.5, "created_at": "2026-06-01T12:30:00Z",
       "merchant_pan_entry_method": "chip_contactless", "digitalization_type": "apple_pay"}'
```

```json
{"transaction_id":null,"minor_category":"restaurants","major_category":"restaurants_bars_cafes","confidence":0.9527931024916834,"top_k":[{"label":"restaurants","probability":0.9527931024916834},{"label":"transport_means","probability":0.016257078203837688},{"label":"supermarkets","probability":0.009271843479240193}],"model_version":"linear_svc-20261018-154858"}
```

Errors: invalid input gives **422** with a list-shaped `detail` (`type`, `loc`, `msg`, sometimes `ctx`), for example `{"detail":[{"type":"missing","loc":["body","amount_value"],"msg":"Field required"}]}`; request bodies are never logged or echoed. A missing model gives **503** `{"detail":"model not loaded"}` on the model routes. A model file that exists but cannot be loaded makes the service fail at startup instead of serving half-broken.

Operational notes:

- The model is loaded once at startup (single process, CPU only); `/health` is a readiness check, not a deep probe.
- `model.joblib` is a pickle, so load only artifacts you trust.
- The three returned probabilities are the top 3 of 12 and need not sum to 1.
- `major_category` is `null` for `refund` (the taxonomy has no major for it).

## Approach and key design choices

- **Predict minor, derive major.** Minor determines major in every row (at most one major per minor), so a second model would only add inconsistent pairs. Major accuracy on the test period is 0.964 (guarded, i.e. with the refund rule applied after prediction).
- **Refund rule.** All 123 refund rows have `status == "refunded"` and `amount_value > 0`, and no other row does. The rule is `status == "refunded"` or `amount_value > 0`; either condition alone triggers it. The model still learns the 12 classes; the rule overrides its probabilities after `predict_proba`. It is valid only if `status` and the sign are known at transaction time (an assumption, see below).
- **Text.** Titles are lower-cased, accent-stripped, digits and punctuation removed. Features are char_wb (character n-grams inside word boundaries) 2-5-gram and word 1-2-gram TF-IDF (sublinear, `min_df=2`). Char n-grams matter because spellings drift (McDonald's alone has 14 normalised variants) and unseen merchants still share sub-strings with seen ones.
- **Features kept:** entry method, digitalization type, recurring frequency, status; `log1p(|amount|)`, sign, fees, weekday, hour, weekend, round-amount flag. **Dropped:** `card_name` (mutual information 0.0286 nats, i.e. natural-log units, vs a 0.0261 permutation baseline, i.e. chance), identifiers and PII (`transaction_id`, `ref_id`, `member_*`), constants, and `merchant_id` (absent for new merchants).
- **Deduplication before splitting:** 120 exact repeats (everything but `transaction_id`) are removed, 8,001 rows remain.
- **Validation.** Temporal split: the first 80 % by `created_at` trains (6,400 rows); the last 1,601 are the test period, starting 2026-05-24. Model selection uses only 5-fold stratified CV on the training period. A separate 5-fold StratifiedGroupKFold grouped by normalised title (no title string appears in both the training and the held-out fold) gives the cold-start view; it is reported, never used for selection. The holdout was inspected during EDA (see assumptions).
- **Selection rule:** CV accuracy, then CV macro-F1, then CV log-loss, then model name; non-finite scores rank last.
- **Model zoo.** Majority class, title lookup, logistic regression, calibrated linear SVC, and a text-only SVC as ablation. A linear model on sparse n-grams is the standard strong baseline for short merchant strings, trains in seconds and is easy to inspect, so it comes first.
- **One artifact, shared preprocessing.** One artifact holds the whole pipeline plus metadata, so serving cannot drift from training. Every stage shares `src/preprocessing.py`.

### What the EDA changed

Details and printed outputs are in `notebooks/eda.ipynb`; notebook section in brackets.

1. **Nulls and constants (2):** `merchant_name` is null on 113 rows (all `RETRAIT DAB`), recurrence on 97.3 %; currency and card type are constant. -> use `title` as text, fill recurrence with `none`, drop constants.
2. **Duplicates (2):** 120 extra copies; 87 of 101 duplicate groups in the training period would straddle CV folds. -> deduplicate before splitting.
3. **Class balance (3):** 12 minor / 8 major classes, largest 29.3 %, max/min ratio 20.1. -> report macro-F1 and per-class recall.
4. **Taxonomy (3):** minor -> major is a strict function; refund (123 rows) has no major; every refund row, and no other, has `status == "refunded"` and `amount > 0`. -> derive major, rule for refunds.
5. **Title noise (4):** 960 raw titles collapse to 875 normalised ones; 31 rows (digit-only titles, 30 of them ATM) normalise to the empty string. -> char n-grams; empty titles use structured features.
6. **Label agreement (4):** 0.9468 of rows carry their title's majority label (0.9612 with refunds set by the rule; all rows, in-sample); 145 of 875 titles carry several labels, covering 73.1 % of rows (`apple com bill`: multimedia 1,048, betting 26, others_shopping 20, refund 16). -> a diagnostic, not a ceiling; part of the error is label noise.
7. **Cold start (7):** 78 of 1,601 test rows (4.9 %) have a title not in training. -> report seen and unseen separately; add title-grouped CV.
8. **Categorical signal (6):** mutual information vs permutation baseline: entry method 0.631 vs 0.004, digitalization 0.167 vs 0.010, status 0.076 vs 0.001, recurrence 0.037 vs 0.001, `card_name` 0.029 vs 0.026. -> keep the first four, drop `card_name`.
9. **Amounts (5):** debits reach -399,180.56, credits 76,098.48 (65 rows, 0.8 %, at or above 10,000 in absolute value); 99.3 % of non-zero ATM amounts in the training period are multiples of 10. -> `log1p(|amount|)` unclipped, sign and round-amount flag as features.
10. **Time (7) and identifiers (8):** weekday and class are dependent (chi-square p = 5.8e-08) but the effect is modest; 91.3 % of test members were seen in training. -> time features in; member and merchant identifiers out.

## Results

| model | cv_accuracy | coldstart_accuracy | test_accuracy (raw) | test_accuracy_guarded | test_macro_f1_guarded | seen_accuracy | unseen_accuracy |
|---|---|---|---|---|---|---|---|
| **linear_svc (selected)** | 0.9403 | 0.7989 | 0.9538 | 0.9550 | 0.9413 | 0.9632 | 0.7949 |
| logreg | 0.9364 | 0.7912 | 0.9525 | 0.9525 | 0.9396 | 0.9613 | 0.7821 |
| linear_svc_text (ablation) | 0.9189 | 0.7284 | 0.9300 | 0.9469 | 0.9320 | 0.9573 | 0.7436 |
| merchant_lookup | 0.8884 | 0.3038 | 0.9094 | 0.9263 | 0.8961 | 0.9626 | 0.2179 |
| majority | 0.2892 | 0.3038 | 0.3061 | 0.3229 | 0.1229 | 0.3283 | 0.2179 |

Legend (from `metrics.json`; "guarded" figures include the refund rule, which is the served behaviour; "raw" is the model alone): `cv_*` is raw model output on the training period; `test_accuracy` is raw output on the test period; `test_*_guarded`, `coldstart_*`, `seen_accuracy` and `unseen_accuracy` have the refund rule applied. The majority class (29.3 % of the data) is the floor.

- **Model vs title lookup.** The title lookup is as good as the model on seen titles (0.9626 vs 0.9632) and collapses on unseen ones (0.2179, no better than the majority class there). The model's gain is generalisation to new strings: 2.9 points of guarded test accuracy overall (0.9550 vs 0.9263).
- **Structured features.** Adding them to the text-only ablation gains 2.1 CV points (0.9403 vs 0.9189) and 7.1 cold-start points. The raw test gap (2.4 points) shrinks to 0.8 once the rule supplies refund detection.
- **Logreg vs SVC** are within noise: 2 test rows apart raw and 4 guarded. The SVC wins on the CV rule; logreg has the better log-loss (CV 0.3008 vs 0.3078, test 0.2251 vs 0.2476), which matters for thresholds.
- **Seen vs unseen.** Seen titles: 0.9632 (1,467 of 1,523; interval 0.953 to 0.972); unseen: 0.7949 (62 of 78; interval 0.692 to 0.870).
- **Confidence and coverage** (guarded, test period, from `evaluation.json`):

| Threshold on confidence | Covered | Accuracy on covered |
|---|---|---|
| 0.80 | 1,515 of 1,601 (94.6 %) | 0.9657 |
| 0.90 | 1,425 of 1,601 (89.0 %) | 0.9663 |
| 0.95 | 183 of 1,601 (11.4 %) | 0.9727 |

Accuracy on covered rows barely moves up to 0.90, then coverage collapses at 0.95. The calibrated SVC rarely reports very high confidence. In the exploratory grid, 0.50 is the smallest threshold reaching 96 % (1,580 covered, 1,521 correct, 21 abstain). It is read off the test set for illustration only; the threshold is a product decision and should be chosen on validation data.

**Errors.** 72 of 1,601 test rows are wrong (guarded). `evaluation.json` assigns each to the first matching bucket (listed in matching order):

| Bucket | Definition | n | share of errors |
|---|---|---|---|
| refund_rule | the rule is exact; no residual errors | 0 | 0 % |
| cold_start | normalised title never seen in training | 16 | 22.2 % |
| thin_support | title seen fewer than 5 times in training | 12 | 16.7 % |
| title_majority | prediction is one of the title's most frequent training labels (title has 5 or more training rows); suspected, not verified, label inconsistency | 44 | 61.1 % |
| other | none of the above; no residual errors | 0 | 0 % |

Examples (title, true -> predicted, confidence): cold_start `LE DIPLOMATE` supermarkets -> restaurants (0.91); thin_support `SODEXO WEB` supermarkets -> restaurants (0.89); title_majority `MC DONALD'S` transport_means -> restaurants (0.95).

- **Weak classes.** `others_shopping` (recall 0.679, precision 0.905, 28 rows) and `betting` (recall 0.885, precision 0.951, 87 rows).
- **Top confusions.** The largest is `apple com bill` true betting -> predicted multimedia (4 rows), then supermarket/restaurant/shopping swaps on big chains. Most look like the same merchant carrying several labels (suspected, not verified); relabelling would be the fix.
- **Normalisation collision.** `G7` normalises to `g` and shares it with other titles; the test row `G7` (true transport_means) is predicted supermarkets (confidence 0.72).
- **Cold-start caveats.** Unseen strings are not unseen merchants (spelling variants can sit in different folds); hyper-parameters were chosen on the whole training period; the fold standard deviation (0.084) is spread, not a confidence interval.
- **Test-set size.** 1,601 rows give about +-1 point, the 78 unseen rows about +-9. The intervals assume independent rows, so they are somewhat optimistic.

## Assumptions and trade-offs

- **Synthetic data.** Real label noise and merchant mix will differ, so the figures above are a baseline, not a forecast.
- **Transaction-time availability.** `status` and recurrence are assumed to be known at transaction time (the data dictionary lists them as such), which lets the refund rule and the recurrence feature work. If `status` were a later update, refunds would have to be predicted from text and sign alone.
- **Recurrence encoding.** Missing `pfm_recurring_frequency` is encoded as `none` and read as non-recurring; that is an assumption, not documented semantics.
- **Label noise is inferred** (same merchant, different labels), not verified; hence suspected inconsistencies are reported, not corrected.
- **UTC.** Hour and weekday are UTC; local-time effects are not modelled.
- **Digit-only titles** normalise to the empty string (31 rows); those rows rely on the structured features, at the cost of weaker text evidence.
- **Holdout inspected.** I looked at the full frame, including the test period, during EDA, so the holdout is not pristine; model and hyper-parameter selection used CV on the training period only.
- **Single window.** One three-month window (2026-03-13 to 2026-06-10): no seasonality, no drift estimate.
- **Calibrated SVC vs logistic regression.** Selection follows accuracy, so the SVC wins by two to four rows while giving up log-loss; a product that thresholds on confidence may prefer logreg.
- **No reliability diagram yet:** the calibration is sigmoid by construction, not verified by plot.

## How I imagined the rest

- A merchant-id cache of confirmed labels in front of the model (merchant ids are unavailable for new merchants, so the model stays as fallback).
- Relabel the high-volume mixed-label titles (`apple com bill`, `intermarche`, `carrefour city`); this is where most of the remaining error appears to sit.
- Per-class thresholds and routing of low-confidence rows to human review; plot a reliability diagram before choosing any threshold.
- Monitoring: share of unseen titles, class mix, confidence drift, alerts on each.
- A cold-start model (char n-grams plus a small multilingual embedding), benchmarked on the title-grouped CV.
- A model registry, and CI that runs the tests and a `--quick` training run.
- Containerisation (not done here).

## Reproducibility

- Seed 42 everywhere. Scores, selected parameters and predictions are deterministic for the pinned versions; timings, timestamps and the `model_version` string (it embeds the training time) are not.
- The artifact is committed, so the API runs without retraining. `python -m src.evaluate` re-checks it against the data (MD5, split date, row counts) and reproduces the stored accuracy (0.9550 guarded).
