# Transaction labeling — design note

Date: 2026-10-18. Author: Zihao Guo.

## Problem and scope

Classify a card transaction into a `minor_category` (12 classes) and its `major_category`
(8 classes), expose the model behind a small API, and explain the work in a README. This note
covers EDA, preprocessing, model training and model evaluation; the API and the README follow.
The code written here keeps that easy: one artifact and one `predict` entry point.

The grading criteria are Methodology (framing, assumptions, baseline and iteration), Exploration
(analysis prior to training), Engineering (code quality, reproducibility, clear run instructions),
API (production concepts) and Communication (concise README with decisions and trade-offs).

## Goals

- Reproducible training from one command producing one artifact (`models/model.joblib`) that
  carries the full sklearn Pipeline (derive → features → classifier) and metadata.
- Clear baseline → iteration story: majority class, merchant lookup, logistic regression,
  calibrated linear SVC, plus a text-only ablation of the SVC. Selection on cross-validation only;
  the temporal test set is reported once.
- Two generalisation views reported side by side: next-period accuracy (temporal split) and
  cold-start accuracy (title-grouped CV, no merchant string shared between folds).
- Error analysis that buckets every test error (refund rule / cold start / thin support / label
  noise / other) and a confidence–coverage curve for a human-review routing story.
- Tests that run in under two minutes (`pytest`), using `--quick` training.

## Non-goals

- API, Dockerfile, README prose, robustness statistics (McNemar, repeated CV).
- Hyper-parameter search beyond the small grids in decision 5; deep-learning or embedding models.
- Any feature built from `member_id`, `member_name`, `transaction_id`, `ref_id`, `merchant_id`.
- Selecting a model from the test split.
- Fitting any preprocessing outside the Pipeline (it would leak across CV folds).

## Data facts that drive the design

| fact | value |
|---|---|
| rows × columns | 8121 × 19; `transaction_id` unique; 120 rows repeat another row on every column except `transaction_id` |
| `minor → major` | strict function; `refund` rows (123) have null `major_category` |
| refund rule | `refund` ⇔ `status == "refunded"` ⇔ `amount_value > 0`, all 123 rows |
| `title == merchant_name` | 98.6 %; `merchant_name` null on 113 rows (title present) |
| constant columns | `amount_currency` (EUR), `card_type` (lydia-card) |
| `pfm_recurring_frequency` | null on 7899 rows, otherwise `monthly` |
| label noise | majority label per normalised title agrees with 94.7 % of non-refund rows in-sample (e.g. `APPLE.COM/BILL`: 1048 multimedia / 26 betting / 20 others_shopping) |
| cold start | 959 distinct `merchant_name`, 389 appear once; 2789 distinct `merchant_id` |
| class balance | supermarkets 2376 … hairdresser 118 |
| period | 2026-03-13 → 2026-06-10 |

## Decisions

1. **Predict minor, derive major** through `MINOR_TO_MAJOR`; `refund` → `None`. Minor and major
   can never disagree.
2. **Refund is a deterministic guard** (`status == "refunded" or amount_value > 0`) applied after
   `predict_proba` in evaluation and serving. The model still learns all 12 classes. Headline test
   metrics are reported with the guard (served behaviour); raw numbers also stored.
3. **Text = `title`**, falling back to `merchant_name`; normalisation: NFKD accent strip, lowercase,
   digits and punctuation → space, collapse whitespace. Features: char_wb 2–5 TF-IDF + word 1–2
   TF-IDF on the normalised text; one-hot on 4 categoricals (`min_frequency=5`, unknown ignored);
   8 scaled numerics derived from amount, fees, timestamp. `card_name` and identifiers unused.
4. **Splits:** temporal 80/20 by `created_at` (test = last period); 5-fold stratified CV on the
   training period for selection; 5-fold `StratifiedGroupKFold` grouped by normalised title on the
   training period for the cold-start metric. Exact duplicates removed before splitting.
5. **Model zoo (5):** `majority`, `merchant_lookup`, `logreg` (grid C ∈ {1,10,30} × class_weight
   ∈ {None, balanced}), `linear_svc` (sigmoid-calibrated LinearSVC, grid C ∈ {0.1,0.5,1.0}),
   `linear_svc_text` (the same SVC on title TF-IDF only: an ablation that isolates the structured
   features). Selection key: CV accuracy, then CV macro-F1, then CV log-loss, then name, scores
   rounded to 12 decimals; a candidate with any non-finite CV score ranks last. Estimator classes
   live in `src/models.py` so that pickled models never reference `__main__`.
6. **Error buckets** (first match wins): `refund_rule` (true or predicted class is refund),
   `cold_start` (normalised title absent from training), `thin_support` (title has fewer than 5
   training rows), `title_majority` (the prediction is one of the labels tied for the highest training
   count of that title: suspected, not verified, label inconsistency), `other`.
7. **Artifacts:** `models/model.joblib` = `{"pipeline": Pipeline, "metadata": dict}`;
   `models/metrics.json`, `models/model_card.json`, `models/model_comparison.csv`;
   `reports/model_comparison.md`, `reports/error_analysis_tables.md`,
   `reports/figures/{confusion_minor,confusion_major,confidence_coverage}.png`.
   Trained artifacts and executed notebook outputs are committed.
8. **Cold-start estimate:** 5-fold StratifiedGroupKFold grouped by normalised title on the
   training period, pooled out-of-fold predictions with the refund rule applied (served behaviour,
   like the seen/unseen split), reported next to the stratified CV for every model, never used for
   selection.

## Repository layout and commands

```
README.md
requirements.txt
Makefile
pytest.ini
data/transactions.csv
notebooks/eda.ipynb
src/{__init__,preprocessing,models,train,evaluate,predict}.py
api/main.py            (later)
models/{model.joblib,metrics.json,model_card.json,model_comparison.csv,evaluation.json}
reports/{eda_findings.md,model_comparison.md,error_analysis_tables.md,error_analysis.md,test_errors.csv,figures/}
tests/
docs/design.md
```

Commands: `make setup`, `make eda`, `make train`, `make evaluate`, `make test`, and
`python -m src.train --quick` for a fast subsampled run (same temporal cut, all 12 classes kept).

## Reproducibility constraints

- Python 3.11 with the versions pinned in `requirements.txt`, notably pandas 3.0.6, numpy 2.4.6,
  scikit-learn 1.9.1 and joblib 1.6.0.
- Seed 42 everywhere (`SEED` in `src/preprocessing.py`).
- `src` is a package, run from the repository root (`python -m src.train`, `python -m src.predict`,
  `pytest`); modules import each other as `from src.preprocessing import ...`.
- The joblib artifact pickles a reference to `src.preprocessing.add_derived_features`, so that
  import path must stay stable.
- The dataset is committed (`data/transactions.csv`) so training is reproducible as-is.
