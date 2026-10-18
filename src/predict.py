"""Dict-in / dict-out predictor: loads the model bundle, validates records, applies the refund guard."""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from src.evaluate import apply_refund_guard
from src.preprocessing import MODEL_PATH, derive_major, frame_from_records


@dataclass
class ModelBundle:
    pipeline: object
    metadata: dict


def load_bundle(path: Path | str = MODEL_PATH) -> ModelBundle:
    # SECURITY: joblib unpickles arbitrary executable content; only load trusted artifacts.
    obj = joblib.load(path)
    pipeline = obj["pipeline"]
    if "refund" not in [str(c) for c in pipeline.classes_]:
        raise ValueError("model artifact must include the required 'refund' class")
    return ModelBundle(pipeline=pipeline, metadata=dict(obj["metadata"]))


def _non_blank(v) -> bool:
    return isinstance(v, str) and v.strip() != ""


_DECIMAL_RE = re.compile(r"\s*[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?\s*", re.ASCII)


def coerce_amount(value) -> float:
    """Convert an amount to a finite float; raise ValueError for anything else."""
    if value is None or isinstance(value, bool):
        raise ValueError("'amount_value' is required and must be a number")
    if isinstance(value, str):
        if not _DECIMAL_RE.fullmatch(value):
            raise ValueError("'amount_value' must be a plain decimal number")
    elif not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError("'amount_value' must be a number")
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("'amount_value' must be a finite number") from exc
    if not math.isfinite(out):
        raise ValueError("'amount_value' must be a finite number")
    return out


def validate_record(record: dict) -> None:
    """Raise ValueError unless the record has usable text and a finite numeric amount."""
    if not isinstance(record, dict):
        raise ValueError("record must be a dict")
    if not (_non_blank(record.get("title")) or _non_blank(record.get("merchant_name"))):
        raise ValueError("record needs a non-blank 'title' or 'merchant_name'")
    coerce_amount(record.get("amount_value"))


def _normalised(record: dict) -> dict:
    out = dict(record)
    out["amount_value"] = coerce_amount(record.get("amount_value"))
    for key in ("title", "merchant_name"):
        if not _non_blank(out.get(key)):
            out[key] = None
    return out


class Predictor:
    def __init__(self, bundle: ModelBundle, top_k: int = 3):
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be an integer >= 1")
        self.bundle = bundle
        self.top_k = top_k
        self.classes_: list[str] = [str(c) for c in bundle.pipeline.classes_]

    @property
    def version(self) -> str:
        return str(self.bundle.metadata.get("model_version", "unknown"))

    @property
    def metadata(self) -> dict:
        return self.bundle.metadata

    def predict(self, records: list[dict]) -> list[dict]:
        records = list(records)
        if not records:
            return []
        for r in records:
            validate_record(r)
        frame = frame_from_records([_normalised(r) for r in records])
        proba = apply_refund_guard(self.bundle.pipeline.predict_proba(frame), frame, self.classes_)
        outputs = []
        for record, p in zip(records, proba, strict=True):
            order = np.argsort(-p, kind="stable")[: self.top_k]
            label = self.classes_[order[0]]
            tid = record.get("transaction_id")
            outputs.append(
                {
                    "transaction_id": None if tid is None else str(tid),
                    "minor_category": label,
                    "major_category": derive_major(label),
                    "confidence": float(p[order[0]]),
                    "top_k": [{"label": self.classes_[i], "probability": float(p[i])} for i in order],
                    "model_version": self.version,
                }
            )
        return outputs

    def predict_one(self, record: dict) -> dict:
        return self.predict([record])[0]


_ARG_TO_COLUMN = {
    "title": "title",
    "merchant_name": "merchant_name",
    "status": "status",
    "created_at": "created_at",
    "entry_method": "merchant_pan_entry_method",
    "digitalization": "digitalization_type",
    "recurring": "pfm_recurring_frequency",
    "fees": "fees",
}


def _record_from_args(args: argparse.Namespace) -> dict:
    record = {"amount_value": args.amount}
    for arg, col in _ARG_TO_COLUMN.items():
        value = getattr(args, arg)
        if value is not None:
            record[col] = value
    return record


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Predict the category of one or more transactions.")
    parser.add_argument("--json", dest="json_input", help="a JSON object or a JSON list of objects")
    parser.add_argument("--title")
    parser.add_argument("--amount", type=float)
    parser.add_argument("--merchant-name")
    parser.add_argument("--status")
    parser.add_argument("--created-at")
    parser.add_argument("--entry-method")
    parser.add_argument("--digitalization")
    parser.add_argument("--recurring")
    parser.add_argument("--fees", type=float)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    args = parser.parse_args(argv)

    if args.json_input is not None:
        try:
            payload = json.loads(args.json_input)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"invalid JSON: {exc}") from exc
        single = not isinstance(payload, list)
        records = [payload] if single else payload
        if not all(isinstance(r, dict) for r in records):
            raise SystemExit("--json must be an object or a list of objects")
    else:
        records, single = [_record_from_args(args)], True
    try:
        out = Predictor(load_bundle(args.model)).predict(records)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(out[0] if single else out, indent=2))


if __name__ == "__main__":
    main()
