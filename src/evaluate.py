"""Evaluation helpers: metrics, refund guard, seen/unseen breakdown, error buckets and report utilities.

Imported by `src.train`; this module never imports `src.train`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)

from src.preprocessing import (
    DATA_PATH,
    MAJOR_CLASSES,
    MODEL_PATH,
    MODELS_DIR,
    REPORTS_DIR,
    TARGET,
    TEXT_COL,
    TEXT_NORM_COL,
    add_derived_features,
    clean,
    dedupe_transactions,
    derive_major,
    load_raw,
    refund_rule,
    temporal_split,
)

logger = logging.getLogger("src.evaluate")

ERROR_BUCKETS = ["refund_rule", "cold_start", "thin_support", "title_majority", "other"]


def _major_or_none(label: str) -> str:
    major = derive_major(label)
    return "none" if major is None else major


@dataclass(frozen=True)
class EvalResult:
    accuracy: float
    macro_f1: float
    log_loss: float
    major_accuracy: float
    major_macro_f1: float
    n: int

    def as_dict(self) -> dict:
        return asdict(self)


def evaluate_predictions(y_true, y_pred, proba, classes) -> EvalResult:
    """Minor-level and major-level metrics. At major level `refund` (no major) maps to "none"."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    major_true = np.array([_major_or_none(v) for v in y_true])
    major_pred = np.array([_major_or_none(v) for v in y_pred])
    return EvalResult(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        log_loss=float(log_loss(y_true, proba, labels=list(classes))),
        major_accuracy=float(accuracy_score(major_true, major_pred)),
        major_macro_f1=float(f1_score(major_true, major_pred, average="macro", zero_division=0)),
        n=len(y_true),
    )


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n (default 95 %)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def apply_refund_guard(proba, df: pd.DataFrame, classes) -> np.ndarray:
    """Return a COPY of `proba` where rows satisfying the refund rule are one-hot on "refund".

    Rows are matched by position; `df` must be the frame `proba` was computed from.
    """
    classes = list(classes)
    if "refund" not in classes:
        raise ValueError("classes must contain 'refund' to apply the refund guard")
    proba = np.array(proba, dtype=float, copy=True)
    if len(df) != len(proba):
        raise ValueError(f"length mismatch: {len(df)} rows vs {len(proba)} probability rows")
    mask = refund_rule(df)
    proba[mask] = 0.0
    proba[mask, classes.index("refund")] = 1.0
    return proba


def predictions_from_proba(proba, classes) -> np.ndarray:
    return np.asarray(list(classes))[np.asarray(proba).argmax(axis=1)]


def seen_unseen_breakdown(train_df: pd.DataFrame, test_df: pd.DataFrame, y_pred) -> dict:
    """Accuracy on test rows whose normalised title occurs in training vs not."""
    seen_titles = set(add_derived_features(train_df)[TEXT_NORM_COL])
    test_norm = add_derived_features(test_df)[TEXT_NORM_COL]
    seen = test_norm.isin(seen_titles).to_numpy()
    correct = test_df[TARGET].to_numpy() == np.asarray(y_pred)

    def acc(mask: np.ndarray) -> float:
        return float(correct[mask].mean()) if mask.any() else float("nan")

    return {
        "seen_n": int(seen.sum()),
        "seen_accuracy": acc(seen),
        "unseen_n": int((~seen).sum()),
        "unseen_accuracy": acc(~seen),
    }


def title_agreement(df: pd.DataFrame, refund_adjusted: bool = False) -> float:
    """In-sample share of rows whose label equals the majority label of their normalised title.

    With `refund_adjusted`, rows satisfying the refund rule count as agreeing and are excluded from
    the per-title majorities. A diagnostic of label noise per merchant, not an upper bound on test
    accuracy.
    """
    d = add_derived_features(df)
    if refund_adjusted:
        is_refund = pd.Series(refund_rule(d), index=d.index)
    else:
        is_refund = pd.Series(False, index=d.index)
    rest = d[~is_refund]
    if rest.empty:
        agree = 0
    else:
        agree = rest.groupby(TEXT_NORM_COL)[TARGET].agg(lambda s: s.value_counts().iloc[0]).sum()
    return float((agree + is_refund.sum()) / len(d))


def per_class_report(y_true, y_pred) -> pd.DataFrame:
    labels = sorted(set(np.asarray(y_true)) | set(np.asarray(y_pred)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return pd.DataFrame({"class": labels, "precision": p, "recall": r, "f1": f, "support": s})


def top_confusions(test_df: pd.DataFrame, y_pred, n: int = 20) -> pd.DataFrame:
    d = pd.DataFrame(
        {
            TEXT_NORM_COL: add_derived_features(test_df)[TEXT_NORM_COL].to_numpy(),
            "true": test_df[TARGET].to_numpy(),
            "pred": np.asarray(y_pred),
        }
    )
    wrong = d[d["true"] != d["pred"]]
    return (
        wrong.groupby([TEXT_NORM_COL, "true", "pred"]).size().reset_index(name="count")
        .sort_values(["count", TEXT_NORM_COL], ascending=[False, True])
        .head(n)
        .reset_index(drop=True)
    )


DEFAULT_THRESHOLDS = [p / 100 for p in range(50, 100, 5)]  # exact decimals (np.arange drifts)


def confidence_coverage(y_true, proba, classes, thresholds=DEFAULT_THRESHOLDS) -> pd.DataFrame:
    """Accuracy on rows whose top probability >= threshold, and the share of rows covered."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    pred = np.asarray(classes)[proba.argmax(axis=1)]
    conf = proba.max(axis=1)
    rows = []
    for t in thresholds:
        mask = conf >= t
        covered_n = int(mask.sum())
        rows.append(
            {
                "threshold": float(t),
                "coverage": float(mask.mean()),
                "accuracy_on_covered": float((pred[mask] == y_true[mask]).mean()) if mask.any() else float("nan"),
                "covered_n": covered_n,
                "correct_n": int((pred[mask] == y_true[mask]).sum()),
            }
        )
    return pd.DataFrame(rows)


def error_buckets(
    train_df: pd.DataFrame, test_df: pd.DataFrame, y_pred, proba, classes, thin_support: int = 5
) -> pd.DataFrame:
    """One row per test error, assigned to the first matching bucket:

    refund_rule    the refund rule fires, or the true or predicted label is refund
    cold_start     the normalised title never occurs in training (the empty string is a normal key)
    thin_support   the title has fewer than `thin_support` training rows
    title_majority the prediction equals ANY label tied for the title's highest training count (suspected label noise)
    other          everything else
    """
    y_pred = np.asarray(y_pred)
    proba = np.asarray(proba)
    y_true = test_df[TARGET].to_numpy()
    test_norm = add_derived_features(test_df)[TEXT_NORM_COL].to_numpy()
    train_norm = add_derived_features(train_df)[TEXT_NORM_COL].to_numpy()
    train_y = train_df[TARGET].to_numpy()
    rule = refund_rule(test_df)
    confidence = proba.max(axis=1)

    counts = pd.DataFrame({"k": train_norm, "y": train_y}).groupby(["k", "y"]).size().reset_index(name="n")
    counts = counts.sort_values(["k", "n", "y"], ascending=[True, False, True])
    majority = counts.drop_duplicates("k").set_index("k")["y"].to_dict()
    top_n = counts.groupby("k")["n"].transform("max")
    top_labels = counts[counts["n"] == top_n].groupby("k")["y"].apply(lambda s: sorted(s)).to_dict()
    totals = counts.groupby("k")["n"].sum().to_dict()

    titles = test_df[TEXT_COL].to_numpy() if TEXT_COL in test_df.columns else test_norm
    rows = []
    for i in np.flatnonzero(y_true != y_pred):
        key = test_norm[i]
        n_train = int(totals.get(key, 0))
        maj = majority.get(key)
        tops = top_labels.get(key, [])
        if rule[i] or y_true[i] == "refund" or y_pred[i] == "refund":
            bucket = "refund_rule"
        elif n_train == 0:
            bucket = "cold_start"
        elif n_train < thin_support:
            bucket = "thin_support"
        elif y_pred[i] in tops:
            bucket = "title_majority"
        else:
            bucket = "other"
        rows.append(
            {
                TEXT_NORM_COL: key,
                "title": titles[i],
                "true": y_true[i],
                "pred": y_pred[i],
                "confidence": float(confidence[i]),
                "bucket": bucket,
                "train_rows_for_title": n_train,
                "train_majority_label": maj,
                "train_top_labels": "|".join(tops),
            }
        )
    columns = [TEXT_NORM_COL, "title", "true", "pred", "confidence", "bucket",
               "train_rows_for_title", "train_majority_label", "train_top_labels"]
    return pd.DataFrame(rows, columns=columns)


def bucket_summary(errors_df: pd.DataFrame) -> pd.DataFrame:
    """Count and share of errors per bucket; all five buckets are listed even when empty."""
    counts = errors_df["bucket"].value_counts() if len(errors_df) else pd.Series(dtype=int)
    total = int(counts.sum())
    return pd.DataFrame(
        {
            "bucket": ERROR_BUCKETS,
            "n": [int(counts.get(b, 0)) for b in ERROR_BUCKETS],
            "share": [float(counts.get(b, 0) / total) if total else 0.0 for b in ERROR_BUCKETS],
        }
    )


def _pyplot():
    """Import pyplot lazily (headless backend) so inference never loads plotting code."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_confusion(y_true, y_pred, labels, path, title) -> None:
    """Row-normalised confusion matrix with annotated cells, saved as PNG."""
    plt = _pyplot()
    labels = list(labels)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    size = max(5.0, 0.75 * len(labels))
    fig, ax = plt.subplots(figsize=(size + 1, size))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, f"{norm[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, label="share of true class")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_confidence_coverage(cc: pd.DataFrame, path) -> None:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(cc["coverage"], cc["accuracy_on_covered"], marker="o")
    offsets = [(6, -12), (6, 8)]
    k = 0
    for _, r in cc.iterrows():
        if not np.isclose([0.50, 0.70, 0.80, 0.90, 0.95], r["threshold"]).any():
            continue
        label = f"{r['threshold']:.2f}"
        if "covered_n" in cc.columns:
            label += f" (n={int(r['covered_n'])})"
        ax.annotate(label, (r["coverage"], r["accuracy_on_covered"]),
                    textcoords="offset points", xytext=offsets[k % 2], fontsize=7)
        k += 1
    ax.set_xlabel("coverage (share of transactions auto-labelled)")
    ax.set_ylabel("accuracy on auto-labelled")
    ax.set_title("Confidence threshold: coverage vs accuracy (test)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def df_to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    def fmt(v) -> str:
        if isinstance(v, (float, np.floating)):
            return format(v, floatfmt)
        return str(v).replace("|", "\\|")

    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)


def to_json_safe(obj):
    """Recursively make an object strict-JSON safe: numpy -> python, non-finite floats -> None."""
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_json_safe(obj.tolist())
    if isinstance(obj, np.generic):
        return to_json_safe(obj.item())
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return str(obj)


def dump_json(obj, path) -> None:
    Path(path).write_text(json.dumps(to_json_safe(obj), indent=2, allow_nan=False))


# --------------------------------------------------------------------------- operating point


def recommended_threshold(cc: pd.DataFrame, n_total: int, target_accuracy: float = 0.96) -> dict | None:
    """Smallest threshold whose accuracy on covered rows reaches the target (None if none does)."""
    ok = cc[cc["accuracy_on_covered"] >= target_accuracy].sort_values("threshold")
    if ok.empty:
        return None
    r = ok.iloc[0]
    covered_n, correct_n = int(r["covered_n"]), int(r["correct_n"])
    low, high = wilson_interval(correct_n, covered_n)
    return {
        "threshold": float(r["threshold"]),
        "coverage": float(r["coverage"]),
        "accuracy_on_covered": float(r["accuracy_on_covered"]),
        "covered_n": covered_n,
        "correct_n": correct_n,
        "abstain_n": int(n_total) - covered_n,
        "accuracy_low": low,
        "accuracy_high": high,
    }


# --------------------------------------------------------------------------- CLI


def _interval(k: int, n: int) -> dict:
    lo, hi = wilson_interval(int(k), int(n))
    return {"k": int(k), "n": int(n), "estimate": (k / n) if n else float("nan"), "low": lo, "high": hi}


def _fmt_ci(d: dict) -> str:
    return f"{d['estimate']:.4f} [{d['low']:.4f}, {d['high']:.4f}] (n={d['n']})"


def _check(problems: list[str], force: bool) -> None:
    for msg in problems:
        if force:
            logger.warning("provenance mismatch (ignored with --force): %s", msg)
    if problems and not force:
        raise SystemExit("provenance check failed (use --force to override):\n  - " + "\n  - ".join(problems))


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Deep-dive evaluation of a trained model on the test period.")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--out", type=Path, default=MODELS_DIR)
    parser.add_argument("--reports", type=Path, default=REPORTS_DIR)
    parser.add_argument("--force", action="store_true",
                        help="evaluate a quick artifact / ignore provenance mismatches")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # SECURITY: joblib unpickles arbitrary code; only load trusted artifacts.
    bundle = joblib.load(args.model)
    pipeline, md = bundle["pipeline"], dict(bundle["metadata"])
    classes = [str(c) for c in pipeline.classes_]
    if "refund" not in classes:
        raise SystemExit("model artifact must include the required 'refund' class")
    if md.get("quick") and not args.force:
        raise SystemExit("refusing to evaluate a quick artifact (metadata quick=True); use --force")

    df = dedupe_transactions(clean(load_raw(args.data)))
    train_df, test_df = temporal_split(df)
    problems = []
    if len(train_df) != md.get("train_rows"):
        problems.append(f"train rows {len(train_df)} != metadata {md.get('train_rows')}")
    if len(test_df) != md.get("test_rows"):
        problems.append(f"test rows {len(test_df)} != metadata {md.get('test_rows')}")
    split_date = test_df["created_at"].min().isoformat()
    if split_date != md.get("split_date"):
        problems.append(f"split_date {split_date} != metadata {md.get('split_date')}")
    md5 = hashlib.md5(Path(args.data).read_bytes()).hexdigest()
    if md5 != md.get("data_md5"):
        problems.append(f"data md5 {md5} != metadata {md.get('data_md5')}")
    _check(problems, args.force)

    y_true = test_df[TARGET].to_numpy()
    proba_raw = np.asarray(pipeline.predict_proba(test_df), dtype=float)
    proba = apply_refund_guard(proba_raw, test_df, classes)
    pred_raw = predictions_from_proba(proba_raw, classes)
    pred = predictions_from_proba(proba, classes)
    ev_raw = evaluate_predictions(y_true, pred_raw, proba_raw, classes)
    ev_g = evaluate_predictions(y_true, pred, proba, classes)
    n = len(test_df)
    su = seen_unseen_breakdown(train_df, test_df, pred)
    intervals = {
        "raw_accuracy": _interval(round(ev_raw.accuracy * n), n),
        "guarded_accuracy": _interval(round(ev_g.accuracy * n), n),
        "seen_accuracy": _interval(round(su["seen_accuracy"] * su["seen_n"]) if su["seen_n"] else 0, su["seen_n"]),
        "unseen_accuracy": _interval(
            round(su["unseen_accuracy"] * su["unseen_n"]) if su["unseen_n"] else 0, su["unseen_n"]),
    }
    per_class = per_class_report(y_true, pred)
    confusions = top_confusions(test_df, pred, n=20)
    errors = error_buckets(train_df, test_df, pred, proba, classes)
    order = errors["bucket"].map({b: i for i, b in enumerate(ERROR_BUCKETS)})
    errors = (errors.assign(_o=order).sort_values(["_o", "confidence"], ascending=[True, False])
              .drop(columns="_o").reset_index(drop=True))
    summary = bucket_summary(errors)
    cc = confidence_coverage(y_true, proba, classes)
    rec = recommended_threshold(cc, n)

    reports, out = Path(args.reports), Path(args.out)
    fig_dir = reports / "figures"
    reports.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    plot_confusion(y_true, pred, classes, fig_dir / "confusion_minor.png", "Minor categories (test, guarded)")
    major = lambda a: [_major_or_none(v) for v in a]  # noqa: E731
    plot_confusion(major(y_true), major(pred), list(MAJOR_CLASSES) + ["none"], fig_dir / "confusion_major.png",
                   "Major categories (test, guarded)")
    plot_confidence_coverage(cc, fig_dir / "confidence_coverage.png")
    errors.to_csv(reports / "test_errors.csv", index=False)

    headline = pd.DataFrame(
        [
            {"variant": "raw (model only)", "accuracy [95% CI]": _fmt_ci(intervals["raw_accuracy"]),
             "macro_f1": ev_raw.macro_f1, "log_loss": ev_raw.log_loss, "major_accuracy": ev_raw.major_accuracy},
            {"variant": "guarded (served)", "accuracy [95% CI]": _fmt_ci(intervals["guarded_accuracy"]),
             "macro_f1": ev_g.macro_f1, "log_loss": ev_g.log_loss, "major_accuracy": ev_g.major_accuracy},
        ]
    )
    seen_tbl = pd.DataFrame(
        [
            {"subset": "seen titles", "accuracy [95% CI]": _fmt_ci(intervals["seen_accuracy"])},
            {"subset": "unseen titles", "accuracy [95% CI]": _fmt_ci(intervals["unseen_accuracy"])},
        ]
    )
    lines = [
        "# Error analysis tables",
        "",
        f"Model `{md.get('model_version')}`; test period starts {split_date} ({n} rows). "
        "Generated by `python -m src.evaluate`; do not edit by hand.",
        "",
        "## Headline",
        "",
        df_to_markdown(headline),
        "",
        "## Per-class report (guarded)",
        "",
        df_to_markdown(per_class),
        "",
        "## Top confusions (guarded)",
        "",
        df_to_markdown(confusions),
        "",
        "## Seen vs unseen titles (guarded)",
        "",
        df_to_markdown(seen_tbl),
        "",
        "## Error buckets (guarded predictions)",
        "",
        df_to_markdown(summary),
        "",
        "`title_majority` errors are suspected label noise (the model agrees with the title's training majority), "
        "not verified noise.",
        "",
    ]
    for b in ERROR_BUCKETS:
        sub = errors[errors["bucket"] == b]
        if sub.empty:
            continue
        lines += [f"### {b}", "", df_to_markdown(sub.head(5)[["title", "true", "pred", "confidence"]]), ""]
    lines += ["## Confidence vs coverage (guarded)", "", df_to_markdown(cc), ""]
    c90 = cc[np.isclose(cc["threshold"], 0.90)]
    c95 = cc[np.isclose(cc["threshold"], 0.95)]
    if len(c90) and len(c95):
        lines += [
            "",
            f"Moving the threshold from 0.90 to 0.95 collapses coverage from {c90.iloc[0]['coverage']:.2%} "
            f"to {c95.iloc[0]['coverage']:.2%}.",
        ]
    lines.append("")
    if rec:
        lines.append(
            f"Exploratory test operating point: threshold {rec['threshold']:.2f} covers "
            f"{rec['covered_n']}/{n} transactions ({rec['coverage']:.2%}); {rec['correct_n']} are correct "
            f"({rec['accuracy_on_covered']:.2%}, Wilson 95 % interval {rec['accuracy_low']:.2%} to "
            f"{rec['accuracy_high']:.2%}); {rec['abstain_n']} abstain. It is the smallest threshold of the "
            "default grid that reaches 96% accuracy; a deployment threshold should be chosen and frozen on "
            "validation data, not on this test set."
        )
    else:
        lines.append("No default threshold reaches 96% accuracy on the covered rows.")
    (reports / "error_analysis_tables.md").write_text("\n".join(lines) + "\n")

    result = {
        "model_name": md.get("model_name"),
        "model_version": md.get("model_version"),
        "data_md5": md5,
        "split_date": split_date,
        "n_test": n,
        "raw": ev_raw.as_dict(),
        "guarded": ev_g.as_dict(),
        "intervals": intervals,
        "seen_unseen": su,
        "per_class": per_class.to_dict("records"),
        "bucket_summary": summary.to_dict("records"),
        "coverage": cc.to_dict("records"),
        "recommended_threshold": rec,
    }
    dump_json(result, out / "evaluation.json")
    print(headline.to_string(index=False))
    return result


if __name__ == "__main__":
    main()
