# Model comparison

- Trained at (UTC): 2026-10-18T15:13:45+00:00  
- Rows after deduplication: 8001 (120 exact duplicates removed)  
- Temporal split: train 6400 rows, test 1601 rows (test starts 2026-05-24T11:07:07+00:00)  
- CV: 5-fold stratified CV on the training period  
- Legend: cv_*: raw model output on the training period (selection); test_accuracy / test_macro_f1 / test_log_loss / test_major_accuracy: raw model output on the test period; *_guarded, coldstart_*, seen_accuracy, unseen_accuracy: with the refund rule applied after prediction (served behaviour; the headline)  
- Cold-start: 5-fold StratifiedGroupKFold grouped by normalised title on the training period (reported only, never used for selection; guarded predictions)  
- Title-agreement diagnostic (in-sample, training period): 0.9464 unadjusted, 0.9602 refund-adjusted (share of rows whose label equals the majority label of their normalised title; a diagnostic, not an upper bound on test accuracy)  
- Selected model: **linear_svc** (selection rule: CV accuracy (desc), then CV macro-F1 (desc), then CV log-loss (asc), then model name (asc); a candidate with any non-finite CV score ranks after every fully valid candidate; scores are rounded to 12 decimals before comparison so float noise never decides a ranking)  
- Selected model test set: n=1601, seen_n=1523, unseen_n=78; guarded test accuracy 0.9550 (Wilson 95% interval 0.9437 to 0.9641)  
- Selection uses raw CV scores; the refund rule helps weak models more than strong ones, so part of the ablation gap is refund detection the rule already provides.

| model | family | cv_accuracy | cv_macro_f1 | cv_log_loss | coldstart_accuracy | coldstart_macro_f1 | coldstart_accuracy_fold_std | test_accuracy | test_macro_f1 | test_log_loss | test_major_accuracy | test_accuracy_guarded | test_macro_f1_guarded | seen_accuracy | unseen_accuracy | search_seconds | refit_seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| linear_svc | linear | 0.9403 | 0.9248 | 0.3078 | 0.7989 | 0.7291 | 0.0837 | 0.9538 | 0.9350 | 0.2476 | 0.9631 | 0.9550 | 0.9413 | 0.9632 | 0.7949 | 89.7185 | 9.9212 |
| logreg | linear | 0.9364 | 0.9196 | 0.3008 | 0.7913 | 0.7225 | 0.0751 | 0.9525 | 0.9396 | 0.2251 | 0.9619 | 0.9525 | 0.9396 | 0.9613 | 0.7821 | 133.7277 | 7.9736 |
| linear_svc_text | ablation | 0.9189 | 0.8311 | 0.4126 | 0.7284 | 0.6195 | 0.0951 | 0.9300 | 0.8414 | 0.3654 | 0.9388 | 0.9469 | 0.9320 | 0.9573 | 0.7436 | 9.0945 | 0.7076 |
| merchant_lookup | baseline | 0.8884 | 0.7923 | 0.7820 | 0.3038 | 0.1211 | 0.0596 | 0.9094 | 0.8086 | 0.6872 | 0.9176 | 0.9263 | 0.8961 | 0.9626 | 0.2179 | 0.2752 | 0.0284 |
| majority | baseline | 0.2892 | 0.0374 | 2.0435 | 0.3038 | 0.1211 | 0.0596 | 0.3061 | 0.0391 | 2.0156 | 0.3061 | 0.3229 | 0.1229 | 0.3283 | 0.2179 | 0.2784 | 0.0253 |

## Best parameters

| model | best_params |
|---|---|
| linear_svc | {"clf__estimator__C": 0.5} |
| logreg | {"clf__C": 10.0, "clf__class_weight": null} |
| linear_svc_text | {"clf__estimator__C": 0.5} |
| merchant_lookup | {} |
| majority | {} |
