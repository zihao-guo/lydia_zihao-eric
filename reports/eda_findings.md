# EDA findings: card-transaction category classifier

Source: executed `notebooks/eda.ipynb`; §n = notebook section.

**Scope.** Descriptive sections (nulls, duplicates, class balance, title agreement) use the full frame. Feature-selection diagnostics (round amounts, categorical signal, mutual information, weekday) use the deduplicated training period (6,400 rows); the full frame, including the holdout, was inspected during exploration. The holdout is therefore not pristine; model and hyper-parameter selection nevertheless used only cross-validation on the training period.

## Data at a glance

8,121 rows × 19 columns, 2026-03-13 → 2026-06-10 (§1, §7). `amount_currency` (EUR) and `card_type` are constant. Nulls (§2): `merchant_name` on 113 rows, all titled `RETRAIT DAB`; `pfm_recurring_frequency` on 7,899 rows (97.27 %), the only value being `monthly` (222); `major_category` on 123 rows, all `refund`, which has no major (§3). Filling null recurrence with `none` is an assumption, not documented semantics.

**Duplicates (§2).** 120 rows repeat another row on every column except `transaction_id` (`ref_id`: 8,001 unique for 8,121 rows). 87 of the 101 training-period pairs would straddle shuffled 5-fold CV; the temporal split has 0 cross-split duplicates. Training rows are deduplicated before split and CV.

## What drives the label

**Merchant text (§4).** 959 non-null raw merchant names, plus the empty string for the 113 null rows; 960 raw titles collapse to 875 normalised. Normalisation merges 38 groups (e.g. `MC DONALD'S` and `MC DONALD S` → `mc donald s`), but distinct spellings remain (`mc donald s` 192, `mcdonald s` 68, `mac donald s` 56; 14 McDonald's variants in all). 31 titles normalise to empty (digit-only strings; 30 sit in the ATM class), so they get no text features and rely on the structured features alone.

**Structural signals** (§5, §6, §7; deduplicated training period, 6,400 rows), with mutual information (MI, nats) against a 20-permutation chance baseline:
- *Refund rule* (full frame, §3). `refund` ⇔ `status == "refunded"` ⇔ `amount_value > 0` on all 123 rows.
- *Entry method*: MI 0.631 vs 0.004 chance (ratio 157) ([entry method](figures/eda_entry_method.png)).
- *Digitalization type*: 0.167 vs 0.010 (ratio 16); `uber` 92 % transport_means, `spotify` 97 % multimedia. `none` covers 100 % of ATM but also 59 % of multimedia, 53 % of tolls and 53 % of others_shopping ([digitalization](figures/eda_digitalization.png)).
- *Status*: 0.076 vs 0.001 (ratio 80). *Recurrence*: 0.037 vs 0.001 (ratio 40); 159 of the 181 training `monthly` rows are multimedia.
- *card_name*: 0.0286 vs 0.0261 chance (ratio 1.10, 30 levels): indistinguishable from noise, so dropped.
- *Fees* > 0 on at most 1.1 % of training rows in any class: a weak signal, but `fees_log` and `has_fees` are kept as cheap model features.
- *Round amounts* (non-zero multiple of €10): 99.3 % of ATM, 28.7 % of betting, ≤ 8.7 % in any other class.
- *Weekday × class* is not independent (χ² p ≈ 5.8e-08) but effects are modest (pharmacies 3.9 % of their rows on Sunday vs 11.5 % overall; weekend share 33.5 % for transport_means and 31.9 % for restaurants).
- *Hour of day*: plotted for three classes only, no test run ([hours](figures/eda_hour_by_class.png)).

**Amounts (§5).** Extremes are disclosed, not clipped: 65 rows (0.8 %) have |amount| ≥ €10,000 (restaurants 36, multimedia 25, supermarkets 2, others_shopping 1, refund 1); the minimum is −€399,180.56 (`MC DONALD S`, restaurants) and the largest credit a €76,098.48 refund. 5 zero-amount ATM rows are excluded from the [amount boxplot](figures/eda_amount_by_class.png).

## Label noise and the title-agreement diagnostic

145 of 875 normalised titles carry more than one label, covering 73.1 % of rows (§4). Labelling every row with its title's majority label agrees **0.9468** in-sample; with refunds solved by the rule first, 0.9612 (315 suspected inconsistencies), and 0.9614 on the deduplicated frame (309 of 8,001). This is a diagnostic yardstick, not an upper bound: it uses only the title, so a model that also sees amount, entry method and digitalization can legitimately exceed it.

Suspected inconsistencies: `apple com bill` (multimedia 1,048 / betting 26 / others_shopping 20 / refund 16), `mc donald s` (restaurants 185 / transport_means 5). Plausibly genuine mixed purchases: Intermarché (supermarkets 324 / restaurants 13 / others_shopping 6 / refund 5) and Amazon (`amazon eu sarl`, 44 rows in others_shopping). Neither reading is verified. Title `RETRAIT DAB` is labelled 109 atm / 4 transport_means; the 4 are also round-amount `chip` rows, so roundness does not separate them.

## Temporal structure and the split

12 complete weeks hold 591–671 transactions (median 620); both edge weeks are partial (255, 306) ([volume](figures/eda_volume_over_time.png), §7). Two splits appear. The raw EDA split (cut 2026-05-24 10:04 UTC, before deduplication) gives 6,496 train / 1,625 test rows, all 12 classes in both; class shares differ by up to 1.74 points (supermarkets 28.9 % → 30.6 %). The final split, after deduplication, gives 6,400 train / 1,601 test rows (test from 2026-05-24 11:07 UTC); the model uses this one. 95.2 % of test rows on the raw split, and 95.1 % (1,523/1,601) on the deduplicated split the model uses, have a *normalised* title that appears in training (raw-title coverage would be lower); on the raw split the 78 unseen rows (70 normalised titles) are mostly restaurants 17, pharmacies 16, supermarkets 14, transport_means 11.

## Leakage and excluded columns

91.3 % of test members were seen in training (§8), so member identity would be memorised. `merchant_id` has 2,789 values (up to 32 per title, 227 with mixed labels) and is unavailable for new merchants. Excluded: `member_id`, `member_name`, `ref_id`, `transaction_id`, `merchant_id`, `amount_currency`, `card_type`, and `card_name` (chance-level MI).

**Assumption (§3).** The test's data dictionary lists `status` and `pfm_recurring_frequency` as available at transaction time, and refunds are separate credit transactions. The refund guard and the recurrence feature depend on that contract.

## Decisions carried into modelling

- **Deduplicate** (120 pairs) before split and CV.
- **Predict minor, derive major** (strict function, §3). The model learns all 12 classes, refund included; the deterministic **refund guard** (123/123) then overrides the model's probabilities after inference, conditional on the availability assumption.
- **`title` text, `merchant_name` fallback**: character + word n-grams (14 McDonald's spellings, 78 unseen test rows).
- **Categoricals**: entry method, digitalization, status, recurrence; `card_name` dropped (1.10× chance). Fees kept as `fees_log` and `has_fees`.
- **Amount**: log1p(|amount|), no capping, sign separate. **Candidates to validate in CV**: round-amount flag, day of week.
- **Temporal split**: the model uses the deduplicated split (6,400 / 1,601; the raw EDA split was 6,496 / 1,625); stratified CV inside train (imbalance 20.1×; [class balance](figures/eda_class_balance.png)).
- **Metrics**: accuracy primary; macro-F1 (top class 29.3 %) and log-loss secondary; minor and major.
- **Yardstick**: 0.9612 agreement as context, not a ceiling.
- **Seen vs unseen** (by normalised title): 95.2 % / 4.8 % of test rows on the raw split (95.1 % / 4.9 % on the deduplicated split the model uses) reported separately.
