import argparse
import json

import joblib
import pytest

from src.models import MODEL_ZOO
from src.predict import Predictor, _record_from_args, coerce_amount, load_bundle, main, validate_record
from src.preprocessing import DATA_PATH, MODEL_PATH, TARGET, clean, load_raw

REC = {"title": "CARREFOUR CITY", "amount_value": -12.5}


@pytest.fixture(scope="module")
def predictor(tiny_bundle_path):
    return Predictor(load_bundle(tiny_bundle_path))


def test_known_merchant(predictor):
    out = predictor.predict_one({**REC, "transaction_id": 7})
    assert out["minor_category"] == "supermarkets" and out["major_category"] == "groceries"
    assert out["transaction_id"] == "7" and out["model_version"] == "test-0"
    assert len(out["top_k"]) == 3 and out["top_k"][0]["label"] == "supermarkets"
    assert out["confidence"] == out["top_k"][0]["probability"]


def test_merchant_name_only(predictor):
    out = predictor.predict_one({"merchant_name": "LA BANQUE POSTALE", "amount_value": -20})
    assert out["minor_category"] == "atm" and out["major_category"] == "finance"
    assert out["transaction_id"] is None


def test_refund_by_amount_and_status(predictor):
    out = predictor.predict_one({**REC, "amount_value": 12.5})
    assert out["minor_category"] == "refund" and out["major_category"] is None and out["confidence"] == 1.0
    out = predictor.predict_one({**REC, "status": " Refunded "})
    assert out["minor_category"] == "refund" and out["major_category"] is None


def test_batch_equals_single_and_empty(predictor):
    recs = [REC, {**REC, "amount_value": 3.0}, {"merchant_name": "LA BANQUE POSTALE", "amount_value": -20}]
    assert predictor.predict(recs) == [predictor.predict_one(r) for r in recs]
    assert predictor.predict([]) == []


@pytest.mark.parametrize(
    "bad",
    [
        {"amount_value": -1},
        {"title": "  ", "amount_value": -1},
        {"title": "X"},
        {"title": "X", "amount_value": float("nan")},
        {"title": "X", "amount_value": float("inf")},
        {"title": "X", "amount_value": True},
        {"title": "X", "amount_value": "abc"},
        {"title": "X", "amount_value": "1_0"},
        {"title": "X", "amount_value": "\u0661\u0660"},
        {"title": "X", "amount_value": 10**400},
    ],
)
def test_invalid_records(predictor, bad):
    with pytest.raises(ValueError):
        validate_record(bad)
    with pytest.raises(ValueError):
        predictor.predict([REC, bad])


def test_numeric_string_amount_ok(predictor):
    validate_record({"title": "X", "amount_value": "-3.5"})


def test_decimal_string_amount(predictor):
    assert coerce_amount("12.5") == 12.5 and coerce_amount(" -1e2 ") == -100.0
    out = predictor.predict_one({**REC, "amount_value": "12.5"})
    assert out == predictor.predict_one({**REC, "amount_value": 12.5})
    assert out["minor_category"] == "refund"


def test_non_string_title_falls_back_to_merchant_name(predictor):
    a = predictor.predict_one({"title": False, "merchant_name": "CARREFOUR CITY", "amount_value": -12.5})
    b = predictor.predict_one({"title": "CARREFOUR CITY", "amount_value": -12.5})
    assert a == b


@pytest.mark.parametrize("k", [0, True, 2.0, "3"])
def test_invalid_top_k(tiny_bundle_path, k):
    with pytest.raises(ValueError):
        Predictor(load_bundle(tiny_bundle_path), top_k=k)


def test_invalid_record_blocks_whole_batch(tiny_bundle_path):
    class Spy:
        def __init__(self, inner):
            self.inner, self.calls = inner, 0
            self.classes_ = inner.classes_

        def predict_proba(self, frame):
            self.calls += 1
            return self.inner.predict_proba(frame)

    bundle = load_bundle(tiny_bundle_path)
    spy = Spy(bundle.pipeline)
    bundle.pipeline = spy
    with pytest.raises(ValueError, match="amount_value"):
        Predictor(bundle).predict([REC, {"title": "X", "amount_value": "abc"}, REC])
    assert spy.calls == 0


def test_cli_invalid_json_and_shape(tiny_bundle_path):
    with pytest.raises(SystemExit, match="invalid JSON"):
        main(["--json", "{bad", "--model", str(tiny_bundle_path)])
    for payload in ("3", '"x"', "[1, 2]"):
        with pytest.raises(SystemExit):
            main(["--json", payload, "--model", str(tiny_bundle_path)])


def test_record_from_args_mapping():
    ns = argparse.Namespace(title="T", amount=-1.5, merchant_name="M", status="completed",
                            created_at="2026-01-01", entry_method="CHIP", digitalization="APPLE_PAY",
                            recurring="MONTHLY", fees=0.5)
    rec = _record_from_args(ns)
    assert rec == {"amount_value": -1.5, "title": "T", "merchant_name": "M", "status": "completed",
                   "created_at": "2026-01-01", "merchant_pan_entry_method": "CHIP",
                   "digitalization_type": "APPLE_PAY", "pfm_recurring_frequency": "MONTHLY", "fees": 0.5}


def test_cli_all_flags(tiny_bundle_path, capsys):
    main(["--title", "CARREFOUR CITY", "--amount", "-12.5", "--merchant-name", "CARREFOUR",
          "--status", "completed", "--created-at", "2026-01-01T10:00:00", "--entry-method", "CHIP",
          "--digitalization", "APPLE_PAY", "--recurring", "MONTHLY", "--fees", "0",
          "--model", str(tiny_bundle_path)])
    out = json.loads(capsys.readouterr().out)
    assert {"transaction_id", "minor_category", "major_category", "confidence", "top_k",
            "model_version"} <= set(out)
    main(["--json", json.dumps({**REC, "transaction_id": "abc"}), "--model", str(tiny_bundle_path)])
    assert json.loads(capsys.readouterr().out)["transaction_id"] == "abc"


def test_bundle_without_refund_rejected(tmp_path):
    df = clean(load_raw(DATA_PATH))
    df = df[df[TARGET] != "refund"].sample(n=500, random_state=0)
    pipe = MODEL_ZOO["merchant_lookup"].factory().fit(df, df[TARGET])
    path = tmp_path / "m.joblib"
    joblib.dump({"pipeline": pipe, "metadata": {}}, path)
    with pytest.raises(ValueError, match="refund"):
        load_bundle(path)


def test_cli_smoke(tiny_bundle_path, capsys):
    main(["--title", "CARREFOUR CITY", "--amount", "-12.5", "--model", str(tiny_bundle_path)])
    out = json.loads(capsys.readouterr().out)
    assert out["minor_category"] == "supermarkets"
    main(["--json", json.dumps([REC, REC]), "--model", str(tiny_bundle_path)])
    assert len(json.loads(capsys.readouterr().out)) == 2


needs_artifact = pytest.mark.skipif(not MODEL_PATH.exists(), reason="models/model.joblib not present")


@needs_artifact
def test_shipped_artifact():
    p = Predictor(load_bundle(MODEL_PATH))
    assert p.predict_one(REC)["minor_category"] == "supermarkets"
    out = p.predict_one({"title": "NETFLIX.COM", "amount_value": -9.99, "merchant_pan_entry_method": "CARD_ON_FILE"})
    assert out["minor_category"] == "multimedia"
    out = p.predict_one({**REC, "status": "refunded"})
    assert out["minor_category"] == "refund" and out["major_category"] is None


def test_predict_import_does_not_load_matplotlib():
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    out = subprocess.run(
        [sys.executable, "-c", "import sys, src.predict; print('matplotlib' in sys.modules)"],
        cwd=root, capture_output=True, text=True, check=True,
    )
    assert out.stdout.strip() == "False"
