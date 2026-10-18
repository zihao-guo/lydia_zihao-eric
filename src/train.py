"""Training CLI: compare the model zoo by CV, score the temporal test period, persist the best model.

Usage: python -m src.train [--models a,b] [--quick] [--out DIR] [--reports DIR] [--data CSV]

Selection uses the training-period CV scores only. The test period is scored once per model and
never influences the choice. A title-grouped (cold-start) CV is reported next to the CV scores but
is also never used for selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold, cross_validate
from threadpoolctl import threadpool_limits

from src import __version__
from src.evaluate import (
    apply_refund_guard,
    df_to_markdown,
    dump_json,
    evaluate_predictions,
    predictions_from_proba,
    seen_unseen_breakdown,
    title_agreement,
    wilson_interval,
)
from src.models import MODEL_ZOO, MerchantLookupClassifier
from src.preprocessing import (
    CAT_COLS,
    DATA_PATH,
    MODELS_DIR,
    NUM_COLS,
    REPORTS_DIR,
    SEED,
    TARGET,
    TEXT_COL,
    TRAIN_FRACTION,
    clean,
    dedupe_transactions,
    load_raw,
    temporal_split,
    title_groups,
)

logger = logging.getLogger("src.train")

QUICK_TRAIN_ROWS = 1200
QUICK_TEST_ROWS = 300

_COMPARISON_COLS = [
    "model", "family", "cv_accuracy", "cv_macro_f1", "cv_log_loss",
    "coldstart_accuracy", "coldstart_macro_f1", "coldstart_accuracy_fold_std",
    "test_accuracy", "test_macro_f1", "test_log_loss", "test_major_accuracy",
    "test_accuracy_guarded", "test_macro_f1_guarded", "seen_accuracy", "unseen_accuracy",
    "search_seconds", "refit_seconds", "best_params",
]

SELECTION_RULE = (
    "CV accuracy (desc), then CV macro-F1 (desc), then CV log-loss (asc), then model name (asc); "
    "a candidate with any non-finite CV score ranks after every fully valid candidate; "
    "scores are rounded to 12 decimals before comparison so float noise never decides a ranking"
)


def selection_key(accuracy: float, macro_f1: float, log_loss_: float) -> tuple:
    """Single ranking key shared by the GridSearchCV refit callable and the final model table
    (ascending sort = best first): (invalid_flag, -accuracy, -macro_f1, log_loss), scores rounded to
    12 decimals. invalid_flag is 1 when any score is non-finite, so a candidate with a higher accuracy
    but a NaN macro-F1 can never beat a fully valid one."""
    values = [float(accuracy), float(macro_f1), float(log_loss_)]
    if not all(np.isfinite(v) for v in values):
        return (1, 0.0, 0.0, 0.0)
    acc, f1, ll = (round(v, 12) for v in values)
    return (0, -acc, -f1, ll)


def select_best_index(cv_results: dict) -> int:
    """GridSearchCV `refit` callable: best candidate by `selection_key`; first wins exact ties."""
    keys = [
        selection_key(a, f, -ll)
        for a, f, ll in zip(
            cv_results["mean_test_accuracy"],
            cv_results["mean_test_macro_f1"],
            cv_results["mean_test_neg_log_loss"],
            strict=False,
        )
    ]
    best = int(min(range(len(keys)), key=lambda i: (keys[i], i)))
    if keys[best][0] == 1:
        raise ValueError("no candidate has finite accuracy, macro-F1 and log-loss")
    return best


def rank_models(rows: list[dict]) -> pd.DataFrame:
    """Comparison table, best first, ordered by `selection_key` then model name."""
    ordered = sorted(
        rows,
        key=lambda r: (*selection_key(r["cv_accuracy"], r["cv_macro_f1"], r["cv_log_loss"]), r["model"]),
    )
    table = pd.DataFrame(ordered).reset_index(drop=True)
    top = ordered[0]
    if selection_key(top["cv_accuracy"], top["cv_macro_f1"], top["cv_log_loss"])[0] == 1:
        raise ValueError("no model has finite CV accuracy, macro-F1 and log-loss")
    return table


def coldstart_cv(train_df, y, groups, factory, best_params, n_splits, seed: int = SEED) -> dict:
    """Title-grouped CV: held-out folds contain only merchant strings the fold's model never saw.

    Predictions are GUARDED (refund rule applied after the model), i.e. cold-start = served behaviour
    on unseen titles, on the same footing as seen/unseen accuracy.

    Metrics are computed once over the pooled out-of-fold predictions (fold sizes are very unequal
    because a few titles hold a large share of the rows), plus the per-fold accuracy std.
    """
    y, groups = np.asarray(y), np.asarray(groups)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pooled = np.empty(len(y), dtype=y.dtype)
    fold_acc = []
    for tr, te in cv.split(train_df, y, groups):
        est = factory().set_params(**best_params)
        est.fit(train_df.iloc[tr], y[tr])
        fold = train_df.iloc[te]
        proba = est.predict_proba(fold)
        classes = list(est.classes_)
        if "refund" in classes:
            proba = apply_refund_guard(proba, fold, classes)
        pooled[te] = predictions_from_proba(proba, classes)
        fold_acc.append(accuracy_score(y[te], pooled[te]))
    return {
        "accuracy": float(accuracy_score(y, pooled)),
        "macro_f1": float(f1_score(y, pooled, average="macro", labels=sorted(set(y)), zero_division=0)),
        "accuracy_fold_std": float(np.std(fold_acc)),
        "n_splits": int(n_splits),
    }


def _describe_features(model) -> dict:
    """Feature description found by introspecting a fitted pipeline: transformer blocks and input columns."""
    names: list[str] = []
    cols: list[str] = []
    steps = getattr(model, "named_steps", {})
    feats = steps.get("features")
    if feats is not None and hasattr(feats, "transformers_"):
        for name, _, columns in feats.transformers_:
            if name == "remainder":
                continue
            names.append(name)
            cols.extend([columns] if isinstance(columns, str) else list(columns))
    if isinstance(steps.get("clf"), MerchantLookupClassifier):
        names.append("title_lookup")
        cols.append("text_norm")
    return {
        "blocks": sorted(set(names), key=names.index),
        "input_columns": sorted(set(cols), key=cols.index),
        "text": "text_norm (normalised title, merchant_name fallback)",
    }


def _stratified_subsample(df: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    """Stratified subsample of about `n_rows` rows; every class keeps at least one row."""
    frac = min(1.0, n_rows / len(df))
    parts = [
        g.sample(n=min(len(g), max(1, round(len(g) * frac))), random_state=SEED)
        for _, g in df.groupby(TARGET, sort=True)
    ]
    out = pd.concat(parts).sort_values("created_at", kind="stable").reset_index(drop=True)
    for label, n in out[TARGET].value_counts().items():
        if n < 6:
            logger.warning("quick mode: class %s has %d rows; calibration with cv=3 may fail", label, n)
    return out


def _md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def run(
    models: list[str] | None = None,
    quick: bool = False,
    out_dir: Path = MODELS_DIR,
    reports_dir: Path = REPORTS_DIR,
    data_path: Path = DATA_PATH,
) -> dict:
    out_dir, reports_dir = Path(out_dir), Path(reports_dir)
    for d in (out_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)
    names = list(models or MODEL_ZOO)
    unknown = [n for n in names if n not in MODEL_ZOO]
    if unknown:
        raise ValueError(f"unknown models {unknown}; available: {list(MODEL_ZOO)}")

    data_md5 = _md5(data_path)
    raw = clean(load_raw(data_path))
    df = dedupe_transactions(raw)  # exact duplicates would leak across CV folds
    rows_deduplicated = len(raw) - len(df)
    train_df, test_df = temporal_split(df)  # split first, so split_date matches a full run
    split_date = test_df["created_at"].min().isoformat()
    if quick:
        train_df = _stratified_subsample(train_df, QUICK_TRAIN_ROWS).reset_index(drop=True)
        test_df = _stratified_subsample(test_df, QUICK_TEST_ROWS).reset_index(drop=True)
    y_train, y_test = train_df[TARGET].to_numpy(), test_df[TARGET].to_numpy()
    groups = title_groups(train_df)
    logger.info("rows=%d (dedup removed %d) train=%d test=%d split=%s quick=%s",
                len(df), rows_deduplicated, len(train_df), len(test_df), split_date, quick)

    n_cv = 2 if quick else 5
    n_cold = 3 if quick else 5
    cv = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=SEED)
    scoring = {
        "accuracy": "accuracy",
        "macro_f1": "f1_macro",
        # labels= lets log_loss score a validation fold that lacks a class; it does not repair a
        # training fold that lacks a class (predict_proba then has fewer columns and the scorer raises)
        "neg_log_loss": make_scorer(
            log_loss, greater_is_better=False, response_method="predict_proba",
            labels=sorted(set(y_train)),
        ),
    }
    agreement = {
        "train": title_agreement(train_df),
        "train_refund_adjusted": title_agreement(train_df, refund_adjusted=True),
    }

    rows, fitted = [], {}
    # small sparse problems + many BLAS threads = spin-wait thrashing
    with threadpool_limits(limits=1, user_api="blas"):
        for name in names:
            spec = MODEL_ZOO[name]
            logger.info("model %s ...", name)
            est = spec.factory()
            t0 = time.perf_counter()
            if spec.param_grid and not quick:
                gs = GridSearchCV(est, spec.param_grid, cv=cv, scoring=scoring,
                                  refit=select_best_index, n_jobs=1)
                gs.fit(train_df, y_train)
                i = gs.best_index_
                cv_acc = gs.cv_results_["mean_test_accuracy"][i]
                cv_f1 = gs.cv_results_["mean_test_macro_f1"][i]
                cv_ll = -gs.cv_results_["mean_test_neg_log_loss"][i]
                best_params, model = dict(gs.best_params_), gs.best_estimator_
                refit_seconds = float(gs.refit_time_)
                search_seconds = time.perf_counter() - t0 - refit_seconds
            else:
                res = cross_validate(est, train_df, y_train, cv=cv, scoring=scoring, n_jobs=1)
                cv_acc, cv_f1 = res["test_accuracy"].mean(), res["test_macro_f1"].mean()
                cv_ll, best_params = -res["test_neg_log_loss"].mean(), {}
                search_seconds = time.perf_counter() - t0
                t1 = time.perf_counter()
                model = est.fit(train_df, y_train)
                refit_seconds = time.perf_counter() - t1

            cold = coldstart_cv(train_df, y_train, groups, spec.factory, best_params, n_cold)

            classes = list(model.classes_)
            proba_raw = model.predict_proba(test_df)
            pred_raw = predictions_from_proba(proba_raw, classes)
            ev_raw = evaluate_predictions(y_test, pred_raw, proba_raw, classes)
            proba_g = apply_refund_guard(proba_raw, test_df, classes)
            pred_g = predictions_from_proba(proba_g, classes)
            ev_g = evaluate_predictions(y_test, pred_g, proba_g, classes)
            su = seen_unseen_breakdown(train_df, test_df, pred_g)
            logger.info("  cv_acc=%.4f coldstart_acc=%.4f test_acc=%.4f (guarded %.4f) "
                        "(search %.1fs, refit %.1fs)",
                        cv_acc, cold["accuracy"], ev_raw.accuracy, ev_g.accuracy,
                        search_seconds, refit_seconds)
            rows.append({
                "model": name, "family": spec.family,
                "cv_accuracy": float(cv_acc), "cv_macro_f1": float(cv_f1), "cv_log_loss": float(cv_ll),
                "coldstart_accuracy": cold["accuracy"], "coldstart_macro_f1": cold["macro_f1"],
                "coldstart_accuracy_fold_std": cold["accuracy_fold_std"],
                "test_accuracy": ev_raw.accuracy, "test_macro_f1": ev_raw.macro_f1,
                "test_log_loss": ev_raw.log_loss, "test_major_accuracy": ev_raw.major_accuracy,
                "test_accuracy_guarded": ev_g.accuracy, "test_macro_f1_guarded": ev_g.macro_f1,
                "seen_accuracy": su["seen_accuracy"], "unseen_accuracy": su["unseen_accuracy"],
                "search_seconds": float(search_seconds), "refit_seconds": float(refit_seconds),
                "best_params": json.dumps(best_params, default=str),
            })
            fitted[name] = {"model": model, "ev_raw": ev_raw, "ev_g": ev_g, "su": su,
                            "best_params": best_params, "cold": cold}

    for r in rows:
        if not all(np.isfinite(r[k]) for k in ("cv_accuracy", "cv_macro_f1", "cv_log_loss")):
            logger.warning("model %s has non-finite CV scores; it is ranked last", r["model"])
    comparison = rank_models(rows)[_COMPARISON_COLS]
    best_name = str(comparison.loc[0, "model"])
    best = fitted[best_name]
    model, best_params = best["model"], best["best_params"]
    classes = list(model.classes_)
    comparison.to_csv(out_dir / "model_comparison.csv", index=False)

    features = _describe_features(model)
    trained_at = datetime.now(UTC)
    model_version = f"{best_name}-{trained_at:%Y%m%d-%H%M%S}"
    best_row = comparison.iloc[0]
    cv_metrics = {"accuracy": float(best_row["cv_accuracy"]), "macro_f1": float(best_row["cv_macro_f1"]),
                  "log_loss": float(best_row["cv_log_loss"])}
    metadata = {
        "model_name": best_name,
        "model_version": model_version,
        "trained_at": trained_at.isoformat(),
        "quick": bool(quick),
        "data_md5": data_md5,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "rows_deduplicated": int(rows_deduplicated),
        "split_date": split_date,
        "classes": [str(c) for c in classes],
        "best_params": best_params,
        "cv_metrics": cv_metrics,
        "coldstart_metrics": best["cold"],
        "test_metrics": best["ev_raw"].as_dict(),
        "test_metrics_guarded": best["ev_g"].as_dict(),
        "seen_unseen": best["su"],
        "title_agreement": agreement,
        "features": features,
        "configured_columns": {"text": TEXT_COL, "categorical": CAT_COLS, "numeric": NUM_COLS},
        "selection_rule": SELECTION_RULE,
        "package_version": __version__,
    }
    joblib.dump({"pipeline": model, "metadata": metadata}, out_dir / "model.joblib")

    metrics = {
        "best_model": best_name,
        "model_version": model_version,
        "quick": bool(quick),
        "data_md5": data_md5,
        "split_date": split_date,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "rows_deduplicated": int(rows_deduplicated),
        "title_agreement": agreement,
        "comparison": comparison.to_dict("records"),
        "selection_rule": SELECTION_RULE,
    }
    dump_json(metrics, out_dir / "metrics.json")

    sub = " (stratified subsample)" if quick else ""
    cv_text = (f"{n_cv}-fold stratified CV on a stratified training subsample (quick mode)" if quick
               else f"{n_cv}-fold stratified CV on the training period")
    model_card = {
        "model_name": best_name,
        "model_version": model_version,
        "trained_at": trained_at.isoformat(),
        "task": "Predict minor_category of a card transaction; major_category is derived from a fixed taxonomy.",
        "classes": metadata["classes"],
        "intended_use": "Auto-labelling of card transactions; use the returned confidence to route "
                        "low-confidence cases to review. Refunds are set by a deterministic rule "
                        "(status refunded or positive amount) applied after the model.",
        "training_data": {
            "source": Path(data_path).name,
            "rows_after_dedup": len(df),
            "rows_deduplicated": int(rows_deduplicated),
            "train_rows": f"{len(train_df)}{sub}" if quick else len(train_df),
            "test_rows": f"{len(test_df)}{sub}" if quick else len(test_df),
            "split": f"temporal, first {TRAIN_FRACTION:.0%} by created_at; test starts {split_date}",
        },
        "model_description": MODEL_ZOO[best_name].description,
        "model_family": MODEL_ZOO[best_name].family,
        "feature_blocks": features["blocks"],
        "input_columns": features["input_columns"],
        "text_feature": features["text"],
        "configured_columns": metadata["configured_columns"],
        "excluded": "card_name and identifier columns are accepted by the API but not used.",
        "selection": {"criterion": f"{SELECTION_RULE}; {cv_text}", "best_params": best_params},
        "metrics": {
            "cv": cv_metrics,
            "coldstart": best["cold"],
            "test": best["ev_raw"].as_dict(),
            "test_guarded": best["ev_g"].as_dict(),
            "seen_unseen": best["su"],
        },
        "diagnostics": {"title_agreement": agreement},
        "limitations": [
            "Labels are noisy per merchant (same merchant carries several labels), so errors on "
            "ambiguous merchants are expected.",
            "Merchants unseen in training are much less accurate; see seen_unseen metrics.",
            "The headline figure is guarded next-period accuracy (refund rule applied) under the current "
            "merchant mix. Cold-start accuracy is also guarded and is measured by title-grouped CV on the "
            "training period (reported in metrics.json)"
            + ("; in this quick run the cold-start CV ran on the training subsample." if quick else "."),
            "Trained on a single three-month window of EUR card transactions; drift is not monitored.",
            "Class imbalance: small classes (e.g. hairdresser) have fewer examples; see per-class report.",
            *(["Quick run (--quick) on a subsample: must not be shipped."] if quick else []),
        ],
        "package_version": __version__,
    }
    dump_json(model_card, out_dir / "model_card.json")

    lo, hi = wilson_interval(round(best["ev_g"].accuracy * len(test_df)), len(test_df))
    su = best["su"]
    header = (
        "# Model comparison\n\n"
        f"- Trained at (UTC): {trained_at.isoformat(timespec='seconds')}  \n"
        f"- Rows after deduplication: {len(df)} ({rows_deduplicated} exact duplicates removed)  \n"
        f"- Temporal split: train {len(train_df)} rows{sub}, test {len(test_df)} rows{sub} "
        f"(test starts {split_date})  \n"
        f"- CV: {cv_text}  \n"
        "- Legend: cv_*: raw model output on the training period (selection); test_accuracy / "
        "test_macro_f1 / test_log_loss / test_major_accuracy: raw model output on the test period; "
        "*_guarded, coldstart_*, seen_accuracy, unseen_accuracy: with the refund rule applied after "
        "prediction (served behaviour; the headline)  \n"
        f"- Cold-start: {n_cold}-fold StratifiedGroupKFold grouped by normalised title on the training "
        "period (reported only, never used for selection; guarded predictions"
        + (", run on the quick-mode training subsample" if quick else "") + ")  \n"
        f"- Title-agreement diagnostic (in-sample, training period): "
        f"{agreement['train']:.4f} unadjusted, {agreement['train_refund_adjusted']:.4f} refund-adjusted "
        "(share of rows whose label equals the majority label of their normalised title; "
        "a diagnostic, not an upper bound on test accuracy)  \n"
        f"- Selected model: **{best_name}** (selection rule: {SELECTION_RULE})  \n"
        f"- Selected model test set: n={len(test_df)}, seen_n={su['seen_n']}, unseen_n={su['unseen_n']}; "
        f"guarded test accuracy {best['ev_g'].accuracy:.4f} (Wilson 95% interval {lo:.4f} to {hi:.4f})  \n"
        "- Selection uses raw CV scores; the refund rule helps weak models more than strong ones, "
        "so part of the ablation gap is refund detection the rule already provides.\n\n"
    )
    (reports_dir / "model_comparison.md").write_text(
        header + df_to_markdown(comparison.drop(columns=["best_params"]))
        + "\n\n## Best parameters\n\n"
        + df_to_markdown(comparison[["model", "best_params"]]) + "\n"
    )
    logger.info("selected %s (%s); artifacts in %s", best_name, model_version, out_dir)
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train and compare transaction category models.")
    parser.add_argument("--models", default=None, help=f"comma-separated subset of {list(MODEL_ZOO)}")
    parser.add_argument("--quick", action="store_true",
                        help="stratified subsample (~1200 train / ~300 test rows), 2-fold CV, no grid search")
    parser.add_argument("--out", type=Path, default=MODELS_DIR)
    parser.add_argument("--reports", type=Path, default=REPORTS_DIR)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    models = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else None
    metrics = run(models=models, quick=args.quick, out_dir=args.out, reports_dir=args.reports,
                  data_path=args.data)
    print(df_to_markdown(pd.DataFrame(metrics["comparison"]).drop(columns=["best_params"])))
    print(f"\nselected: {metrics['best_model']} ({metrics['model_version']})")


if __name__ == "__main__":
    main()
