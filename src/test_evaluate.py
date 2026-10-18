import numpy as np
import pandas as pd
import pytest

from src import evaluate as ev


def _frame(titles, labels, amounts=None, status=None):
    n = len(titles)
    d = {"title": titles, "amount_value": amounts if amounts is not None else [-1.0] * n,
         "fees": [0] * n, "created_at": [None] * n, "minor_category": labels}
    if status is not None:
        d["status"] = status
    return pd.DataFrame(d)


def test_evaluate_predictions_perfect():
    classes = ["atm", "refund", "supermarkets"]
    y = np.array(["atm", "refund", "supermarkets", "atm"])
    proba = np.eye(3)[[0, 1, 2, 0]]
    r = ev.evaluate_predictions(y, y, proba, classes)
    assert r.accuracy == 1.0 and r.macro_f1 == 1.0 and r.major_accuracy == 1.0 and r.n == 4
    assert r.major_macro_f1 == 1.0
    assert r.log_loss < 1e-6
    assert set(r.as_dict()) == {"accuracy", "macro_f1", "log_loss", "major_accuracy", "major_macro_f1", "n"}


def test_major_metrics_map_refund_to_none():
    classes = ["refund", "supermarkets"]
    y_true = np.array(["refund", "supermarkets"])
    y_pred = np.array(["supermarkets", "supermarkets"])
    proba = np.array([[0.4, 0.6], [0.1, 0.9]])
    r = ev.evaluate_predictions(y_true, y_pred, proba, classes)
    assert r.accuracy == 0.5 and r.major_accuracy == 0.5
    # major labels: true [none, groceries], pred [groceries, groceries]
    # F1(none)=0, F1(groceries)=2/3 -> macro 1/3
    assert r.major_macro_f1 == pytest.approx(1 / 3)


def test_major_metrics_merge_minors_of_same_major():
    classes = ["hairdresser", "pharmacies"]
    y_true = np.array(["pharmacies", "hairdresser"])
    y_pred = np.array(["hairdresser", "pharmacies"])
    proba = np.array([[0.6, 0.4], [0.4, 0.6]])
    r = ev.evaluate_predictions(y_true, y_pred, proba, classes)
    assert r.accuracy == 0.0 and r.major_accuracy == 1.0 and r.major_macro_f1 == 1.0


def test_title_agreement_and_seen_unseen():
    train = _frame(["A", "A", "A", "B"], ["x", "x", "y", "x"])
    test = _frame(["A", "C"], ["x", "x"])
    assert ev.title_agreement(train) == 0.75
    refund_row = _frame(["A"], ["refund"], amounts=[5.0], status=["refunded"])
    assert ev.title_agreement(pd.concat([train, refund_row], ignore_index=True), refund_adjusted=True) == 0.8
    out = ev.seen_unseen_breakdown(train, test, np.array(["x", "y"]))
    assert out == {"seen_n": 1, "seen_accuracy": 1.0, "unseen_n": 1, "unseen_accuracy": 0.0}


def test_title_agreement_all_refund_frame():
    df = _frame(["A", "B"], ["refund", "refund"], amounts=[5.0, 6.0], status=["refunded", "refunded"])
    assert ev.title_agreement(df, refund_adjusted=True) == 1.0


def test_confidence_coverage_monotone_coverage():
    classes = ["a", "b"]
    y = np.array(["a", "b", "a", "b"])
    proba = np.array([[0.9, 0.1], [0.3, 0.7], [0.55, 0.45], [0.2, 0.8]])
    df = ev.confidence_coverage(y, proba, classes, thresholds=np.array([0.5, 0.6, 0.8]))
    assert df["coverage"].tolist() == [1.0, 0.75, 0.5]


def test_confidence_coverage_default_thresholds_include_exact_decimal():
    classes = ["a", "b"]
    y = np.array(["a", "a", "b"])
    proba = np.array([[0.8, 0.2], [0.8, 0.2], [0.3, 0.7]])
    df = ev.confidence_coverage(y, proba, classes)
    row = df[np.isclose(df["threshold"], 0.8)].iloc[0]
    assert row["threshold"] == 0.8
    assert np.isclose(row["coverage"], 2 / 3) and row["accuracy_on_covered"] == 1.0


def test_no_plot_or_markdown_helpers():
    for name in ("plot_confusion", "plot_confidence_coverage", "_pyplot", "df_to_markdown"):
        assert not hasattr(ev, name)


def test_evaluate_import_does_not_load_matplotlib():
    import subprocess
    import sys
    from pathlib import Path

    proc = subprocess.run(
        [sys.executable, "-c", "import sys, src.evaluate; print('matplotlib' in sys.modules)"],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False"


def test_per_class_and_top_confusions():
    test = _frame(["A", "A", "B", "B"], ["x", "x", "y", "y"])
    pred = np.array(["x", "y", "y", "y"])
    rep = ev.per_class_report(test["minor_category"], pred)
    assert rep.set_index("class").loc["x", "recall"] == 0.5 and rep["support"].sum() == 4
    conf = ev.top_confusions(test, pred)
    assert conf.to_dict("records") == [{"text_norm": "a", "true": "x", "pred": "y", "count": 1}]


def test_wilson_interval():
    low, high = ev.wilson_interval(0, 50)
    assert low == 0.0 and 0.0 < high < 0.1
    low, high = ev.wilson_interval(50, 50)
    assert high == 1.0 and 0.9 < low < 1.0
    low, high = ev.wilson_interval(50, 100)
    assert (low + high) / 2 == pytest.approx(0.5, abs=1e-9)
    assert low < 0.5 < high
    assert np.isnan(ev.wilson_interval(0, 0)[0])


def test_apply_refund_guard():
    classes = ["atm", "refund", "supermarkets"]
    df = _frame(["a", "b", "c", "d"], ["atm", "refund", "supermarkets", "refund"],
                amounts=[-5.0, 3.0, -2.0, -1.0], status=["completed", "completed", "completed", " Refunded "])
    proba = np.array([[0.7, 0.2, 0.1]] * 4)
    original = proba.copy()
    out = ev.apply_refund_guard(proba, df, classes)
    assert out is not proba
    np.testing.assert_array_equal(proba, original)
    np.testing.assert_array_equal(out[1], [0, 1, 0])
    np.testing.assert_array_equal(out[3], [0, 1, 0])
    np.testing.assert_array_equal(out[0], original[0])
    np.testing.assert_array_equal(out[2], original[2])
    with pytest.raises(ValueError):
        ev.apply_refund_guard(proba, df, ["atm", "supermarkets", "betting"])
    with pytest.raises(ValueError):
        ev.apply_refund_guard(proba[:3], df, classes)


def test_predictions_from_proba():
    out = ev.predictions_from_proba(np.array([[0.1, 0.9], [0.8, 0.2]]), ["a", "b"])
    assert out.tolist() == ["b", "a"]


def _bucket_fixture():
    # training: "a" x6 (x), "b" x2 (x), "" x6 (x:4, y:2), "c" x5 (x:3, z:2)
    titles = ["A"] * 6 + ["B"] * 2 + ["123"] * 6 + ["C"] * 5
    labels = ["x"] * 6 + ["x"] * 2 + ["x"] * 4 + ["y"] * 2 + ["x"] * 3 + ["z"] * 2
    train = _frame(titles, labels)
    test_titles = ["Q", "B", "A", "123", "C", "A", "A"]
    test_labels = ["x", "x", "x", "y", "z", "x", "y"]
    amounts = [-1.0, -1.0, -1.0, -1.0, -1.0, 4.0, -1.0]
    test = _frame(test_titles, test_labels, amounts=amounts)
    pred = np.array(["y", "z", "z", "x", "x", "x", "x"])
    # Q: cold_start; B: thin_support (2 rows); A: other (majority x, pred z);
    # "123" -> "": title_majority (pred x == majority x); C: title_majority (pred x == majority x);
    # row 5 (A, correct, no error); row 6 (A, true y, pred x): title_majority
    proba = np.full((7, 3), 1 / 3)
    proba[:, 0] = 0.5
    return train, test, pred, proba


def test_error_buckets_cover_all_five_buckets():
    train, test, pred, proba = _bucket_fixture()
    # build a refund_rule error: positive amount, predicted something else
    test.loc[5, "minor_category"] = "refund"
    err = ev.error_buckets(train, test, pred, proba, ["x", "y", "z"])
    by_title = {(r.title, r.true): r for r in err.itertuples()}
    assert by_title[("Q", "x")].bucket == "cold_start"
    assert by_title[("B", "x")].bucket == "thin_support"
    assert by_title[("A", "x")].bucket == "other"
    assert by_title[("123", "y")].bucket == "title_majority"
    assert by_title[("123", "y")].text_norm == ""  # empty string is a normal key, not cold start
    assert by_title[("123", "y")].train_rows_for_title == 6
    assert by_title[("123", "y")].train_majority_label == "x"
    assert by_title[("C", "z")].bucket == "title_majority"
    assert by_title[("A", "refund")].bucket == "refund_rule"
    assert len(err) == 7
    assert set(err["bucket"]) == set(ev.ERROR_BUCKETS)
    assert list(err.columns) == ["text_norm", "title", "true", "pred", "confidence", "bucket",
                                 "train_rows_for_title", "train_majority_label", "train_top_labels"]
    assert err["confidence"].between(0, 1).all()


def test_error_buckets_no_errors_and_summary():
    train, test, _, proba = _bucket_fixture()
    err = ev.error_buckets(train, test, test["minor_category"].to_numpy(), proba, ["x", "y", "z"])
    assert err.empty
    summ = ev.bucket_summary(err)
    assert summ["bucket"].tolist() == ev.ERROR_BUCKETS and summ["n"].sum() == 0


def test_bucket_summary_shares_sum_to_one():
    train, test, pred, proba = _bucket_fixture()
    err = ev.error_buckets(train, test, pred, proba, ["x", "y", "z"])
    summ = ev.bucket_summary(err)
    assert summ["bucket"].tolist() == ev.ERROR_BUCKETS
    assert summ["share"].sum() == pytest.approx(1.0)
    assert summ["n"].sum() == len(err)
    empty = ev.bucket_summary(err.iloc[0:0])
    assert len(empty) == 5 and empty["share"].sum() == 0.0


def test_to_json_safe_and_dump_json(tmp_path):
    import json

    out = ev.to_json_safe({"a": float("nan"), "b": np.float64("inf"), "c": [np.int64(3), np.float32(1.5)]})
    assert out == {"a": None, "b": None, "c": [3, 1.5]}
    path = tmp_path / "x.json"
    ev.dump_json({"v": float("nan"), "w": np.arange(2)}, path)
    assert json.loads(path.read_text()) == {"v": None, "w": [0, 1]}


def test_error_buckets_majority_tie_break():
    # title A: x and y tie 2-2 -> displayed majority is "x" (count desc, then label asc); predicting
    # either tied label is title_majority.
    train = _frame(["A"] * 4, ["y", "x", "y", "x"])
    test = _frame(["A", "A"], ["z", "z"])
    pred = np.array(["x", "y"])
    proba = np.full((2, 3), 1 / 3)
    err = ev.error_buckets(train, test, pred, proba, ["x", "y", "z"], thin_support=2)
    assert err["train_majority_label"].tolist() == ["x", "x"]
    assert err["train_top_labels"].tolist() == ["x|y", "x|y"]
    assert err["bucket"].tolist() == ["title_majority", "title_majority"]


def test_error_buckets_tie_later_label_and_thin_boundary():
    train = _frame(["A"] * 6 + ["B"] * 5, ["x"] * 3 + ["y"] * 3 + ["x"] * 5)
    test = _frame(["A", "B"], ["z", "z"])
    err = ev.error_buckets(train, test, np.array(["y", "y"]), np.full((2, 3), 1 / 3), ["x", "y", "z"])
    by = err.set_index("title")
    assert by.loc["A", "bucket"] == "title_majority" and by.loc["A", "train_majority_label"] == "x"
    assert by.loc["B", "train_rows_for_title"] == 5 and by.loc["B", "bucket"] == "other"  # 5 rows: not thin


def test_apply_refund_guard_rows_sum_to_one_and_input_untouched():
    classes = ["atm", "refund", "supermarkets"]
    df = _frame(["a", "b"], ["atm", "refund"], amounts=[-1.0, 2.0])
    proba = np.array([[0.5, 0.3, 0.2], [0.6, 0.2, 0.2]])
    original = proba.copy()
    out = ev.apply_refund_guard(proba, df, classes)
    np.testing.assert_allclose(out.sum(axis=1), 1.0)
    np.testing.assert_array_equal(proba, original)


def test_recommended_threshold():
    cc = pd.DataFrame({"threshold": [0.5, 0.6, 0.7, 0.8],
                       "coverage": [1.0, 0.9, 0.7, 0.5],
                       "accuracy_on_covered": [0.90, 0.95, 0.97, 0.99],
                       "covered_n": [100, 90, 70, 50],
                       "correct_n": [90, 85, 68, 50]})
    rec = ev.recommended_threshold(cc, 100)
    assert rec["threshold"] == 0.7 and rec["coverage"] == 0.7 and rec["accuracy_on_covered"] == 0.97
    assert (rec["covered_n"], rec["correct_n"], rec["abstain_n"]) == (70, 68, 30)
    assert rec["accuracy_low"] == pytest.approx(ev.wilson_interval(68, 70)[0])
    assert rec["accuracy_high"] == pytest.approx(ev.wilson_interval(68, 70)[1])
    assert ev.recommended_threshold(cc, 100, target_accuracy=0.999) is None
    cc.loc[1, "accuracy_on_covered"] = float("nan")
    assert ev.recommended_threshold(cc, 100)["threshold"] == 0.7


def test_confidence_coverage_counts():
    y = np.array(["a", "b", "a", "b"])
    proba = np.array([[0.9, 0.1], [0.3, 0.7], [0.55, 0.45], [0.8, 0.2]])
    df = ev.confidence_coverage(y, proba, ["a", "b"], thresholds=[0.5, 0.8])
    assert list(df.columns)[:3] == ["threshold", "coverage", "accuracy_on_covered"]
    assert df["covered_n"].tolist() == [4, 2] and df["correct_n"].tolist() == [3, 1]


@pytest.fixture(scope="module")
def quick_run(tmp_path_factory):
    from src.train import run

    base = tmp_path_factory.mktemp("evalrun")
    out = base / "models"
    run(models=["majority", "merchant_lookup"], quick=True, out_dir=out)
    return out


def _argv(out, model=None, *extra):
    return ["--model", str(model or out / "model.joblib"), "--out", str(out), *extra]


def test_evaluate_cli_end_to_end(quick_run, tmp_path, capsys):
    import json

    out = quick_run
    with pytest.raises(SystemExit):
        ev.main(_argv(out))
    out2 = tmp_path / "m"
    ev.main(_argv(out2, out / "model.joblib", "--force"))
    assert [p.name for p in tmp_path.iterdir()] == ["m"]
    assert [p.name for p in out2.iterdir()] == ["evaluation.json"]
    stdout = capsys.readouterr().out
    for text in ("raw (model only)", "guarded (served)", "accuracy [95% CI]", "seen titles", "unseen titles",
                 "bucket", "cold_start", "covered_n", "correct_n", "threshold"):
        assert text in stdout
    res = json.loads((out2 / "evaluation.json").read_text())
    assert {"model_name", "model_version", "data_md5", "split_date", "n_test", "raw", "guarded", "intervals",
            "seen_unseen", "per_class", "top_confusions", "errors", "examples_per_bucket",
            "bucket_summary", "coverage", "recommended_threshold"} <= set(res)
    assert all({"covered_n", "correct_n"} <= set(r) for r in res["coverage"])
    rt = res["recommended_threshold"]
    if rt is not None:
        assert rt["covered_n"] + rt["abstain_n"] == res["n_test"]
    assert [b["bucket"] for b in res["bucket_summary"]] == ev.ERROR_BUCKETS
    assert sum(b["n"] for b in res["bucket_summary"]) == len(res["errors"])
    assert set(res["examples_per_bucket"]) == set(ev.ERROR_BUCKETS)
    assert all(len(v) <= 5 for v in res["examples_per_bucket"].values())
    if res["errors"]:
        assert {"title", "text_norm", "true", "pred", "confidence", "bucket", "train_rows_for_title",
                "train_majority_label", "train_top_labels"} == set(res["errors"][0])


@pytest.mark.parametrize("field,value", [("data_md5", "0" * 32), ("train_rows", -1)])
def test_evaluate_provenance_rejection_and_force(quick_run, tmp_path, field, value):
    import joblib

    out = quick_run
    bundle = joblib.load(out / "model.joblib")
    bundle["metadata"] = {**bundle["metadata"], field: value, "quick": False}
    bad = tmp_path / "bad.joblib"
    joblib.dump(bundle, bad)
    o = tmp_path / "o"
    with pytest.raises(SystemExit):
        ev.main(_argv(o, bad))
    assert not (o / "evaluation.json").exists()
    ev.main(_argv(o, bad, "--force"))
    assert (o / "evaluation.json").exists()


def test_shipped_evaluation_matches_training_metadata():
    import json

    import joblib

    from src.preprocessing import MODELS_DIR

    model, ej = MODELS_DIR / "model.joblib", MODELS_DIR / "evaluation.json"
    if not (model.exists() and ej.exists()):
        pytest.skip("shipped artifact or evaluation.json missing")
    md = joblib.load(model)["metadata"]
    ejd = json.loads(ej.read_text())
    assert ejd["model_version"] == md["model_version"]
    assert ejd["guarded"]["accuracy"] == md["test_metrics_guarded"]["accuracy"]
