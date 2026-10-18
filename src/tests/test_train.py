import json
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from threadpoolctl import threadpool_limits

from src.models import MODEL_ZOO, MerchantLookupClassifier
from src.preprocessing import (
    DATA_PATH,
    REPO_ROOT,
    TARGET,
    clean,
    load_raw,
    temporal_split,
    title_groups,
)
from src.train import _describe_features, _stratified_subsample, coldstart_cv, rank_models, run, select_best_index

FULL_SPLIT_DATE = "2026-05-24T11:07:07+00:00"


def _strict_loads(text):
    def bad(token):
        raise ValueError(f"non-finite JSON constant {token}")

    return json.loads(text, parse_constant=bad)


@pytest.fixture(scope="module")
def quick_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("quick")
    out = root / "models"
    metrics = run(models=["majority", "merchant_lookup"], quick=True, out_dir=out, data_path=DATA_PATH)
    return metrics, out, root


@pytest.fixture(scope="module")
def sample_split():
    df = clean(load_raw(DATA_PATH))
    sample = df.groupby(TARGET, group_keys=False).sample(frac=0.15, random_state=0)
    return temporal_split(sample)


def test_quick_run_writes_artifacts(quick_run):
    metrics, out, root = quick_run
    assert sorted(p.name for p in out.iterdir()) == [
        "metrics.json", "model.joblib", "model_card.json", "model_comparison.csv"]
    assert [p.name for p in root.iterdir()] == ["models"]  # nothing is written outside the model directory
    bundle = joblib.load(out / "model.joblib")
    assert set(bundle) == {"pipeline", "metadata"}
    md = bundle["metadata"]
    table = json.loads((out / "metrics.json").read_text())["comparison"]
    assert metrics["best_model"] == table[0]["model"] == md["model_name"]
    assert {r["model"] for r in table} == {"majority", "merchant_lookup"}
    assert md["quick"] is True
    assert len(md["data_md5"]) == 32 and int(md["data_md5"], 16) >= 0
    assert md["split_date"] == FULL_SPLIT_DATE == metrics["split_date"]
    assert md["train_rows"] < 1500 and md["test_rows"] < 400
    assert md["test_metrics_guarded"]["n"] == md["test_rows"]
    assert md["coldstart_metrics"]["n_splits"] == 3
    saved = json.loads((out / "metrics.json").read_text())
    notes = " ".join(saved["notes"])
    assert "Selected model" in notes and "Wilson 95%" in notes and "seen_n=" in notes
    assert "(stratified subsample)" in notes and "Selection uses raw CV" in notes
    assert "Cold-start" in notes and "CV:" in notes
    assert saved["legend"] and all(isinstance(v, str) for v in saved["legend"].values())
    assert set(md["title_agreement"]) == {"train", "train_refund_adjusted"}


def test_quick_run_json_strict_and_columns(quick_run):
    _, out, _ = quick_run
    metrics = _strict_loads((out / "metrics.json").read_text())
    card = _strict_loads((out / "model_card.json").read_text())
    assert metrics["quick"] is True and "selection_rule" in metrics
    expected = ["model", "family", "cv_accuracy", "cv_macro_f1", "cv_log_loss", "coldstart_accuracy",
                "coldstart_macro_f1", "coldstart_accuracy_fold_std", "test_accuracy", "test_macro_f1",
                "test_log_loss", "test_major_accuracy", "test_accuracy_guarded", "test_macro_f1_guarded",
                "seen_accuracy", "unseen_accuracy", "search_seconds", "refit_seconds", "best_params"]
    for row in metrics["comparison"]:
        assert list(row) == expected
        assert 0.0 <= row["coldstart_accuracy"] <= 1.0
        assert 0.0 <= row["test_accuracy_guarded"] <= 1.0
    assert any("quick" in s.lower() for s in card["limitations"])
    assert any("Cold-start accuracy" in s for s in card["limitations"])
    assert card["model_family"] == "baseline"


def test_artifact_loads_in_fresh_interpreter(quick_run):
    _, out, _ = quick_run
    proc = subprocess.run(
        [sys.executable, "-c", "import joblib,sys; b=joblib.load(sys.argv[1]); print(len(b['pipeline'].classes_))",
         str(out / "model.joblib")],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "12"


def test_zoo_is_exactly_the_five_models():
    assert list(MODEL_ZOO) == ["majority", "merchant_lookup", "logreg", "linear_svc", "linear_svc_text"]
    assert [s.family for s in MODEL_ZOO.values()] == ["baseline", "baseline", "linear", "linear", "ablation"]
    assert MODEL_ZOO["majority"].param_grid is None and MODEL_ZOO["merchant_lookup"].param_grid is None


@pytest.mark.parametrize("name", list(MODEL_ZOO))
def test_zoo_models_fit_predict_proba(name, sample_split):
    train, test = sample_split
    pipe = MODEL_ZOO[name].factory()
    with threadpool_limits(limits=1, user_api="blas"):
        pipe.fit(train, train[TARGET])
    proba = pipe.predict_proba(test)
    assert proba.shape == (len(test), len(pipe.classes_))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert set(pipe.predict(test)) <= set(pipe.classes_)


def test_merchant_lookup_unseen_key_gets_prior(sample_split):
    train, _ = sample_split
    pipe = MODEL_ZOO["merchant_lookup"].factory().fit(train, train[TARGET])
    new = train.head(1).copy()
    new["title"] = "ZZZ UNSEEN MERCHANT"
    proba = pipe.predict_proba(new)[0]
    clf: MerchantLookupClassifier = pipe.named_steps["clf"]
    np.testing.assert_allclose(proba, clf.prior_)


def test_coldstart_cv_pooled_metrics(sample_split):
    train, _ = sample_split
    y = train[TARGET].to_numpy()
    res = coldstart_cv(train, y, title_groups(train), MODEL_ZOO["majority"].factory, {}, n_splits=3)
    assert res["n_splits"] == 3
    assert 0.0 <= res["accuracy"] <= 1.0 and 0.0 <= res["macro_f1"] <= 1.0
    assert res["accuracy_fold_std"] >= 0.0
    assert set(res) == {"accuracy", "macro_f1", "accuracy_fold_std", "n_splits"}


def test_select_best_index_lexicographic_with_first_tie():
    res = {
        "mean_test_accuracy": [0.9, 0.9, 0.9, 0.8],
        "mean_test_macro_f1": [0.5, 0.7, 0.7, 0.99],
        "mean_test_neg_log_loss": [-0.1, -0.5, -0.3, -0.01],
    }
    assert select_best_index(res) == 2
    res["mean_test_neg_log_loss"][2] = -0.5
    assert select_best_index(res) == 1  # exact tie -> first


def test_selection_paths_agree_on_near_ties():
    accs = [0.9399999999999998, 0.9400000000000001]
    f1s = [0.95, 0.90]
    lls = [0.3, 0.3]
    refit = select_best_index({
        "mean_test_accuracy": accs, "mean_test_macro_f1": f1s,
        "mean_test_neg_log_loss": [-v for v in lls],
    })
    rows = [{"model": f"m{i}", "cv_accuracy": accs[i], "cv_macro_f1": f1s[i], "cv_log_loss": lls[i]}
            for i in range(2)]
    assert rank_models(rows).loc[0, "model"] == f"m{refit}" == "m0"
    refit2 = select_best_index({
        "mean_test_accuracy": accs[::-1], "mean_test_macro_f1": f1s[::-1],
        "mean_test_neg_log_loss": [-v for v in lls],
    })
    assert refit2 == 1
    assert rank_models(rows[::-1]).loc[0, "model"] == "m0"


def test_nan_cv_scores_rank_last_in_both_paths():
    nan = float("nan")
    res = {
        "mean_test_accuracy": [nan, 0.914, 0.90],
        "mean_test_macro_f1": [nan, 0.5, 0.6],
        "mean_test_neg_log_loss": [nan, -0.4, -0.3],
    }
    assert select_best_index(res) == 1
    rows = [
        {"model": "broken", "cv_accuracy": nan, "cv_macro_f1": nan, "cv_log_loss": nan},
        {"model": "lookup", "cv_accuracy": 0.914, "cv_macro_f1": 0.5, "cv_log_loss": 0.4},
        {"model": "other", "cv_accuracy": 0.90, "cv_macro_f1": 0.6, "cv_log_loss": 0.3},
    ]
    assert rank_models(rows)["model"].tolist() == ["lookup", "other", "broken"]


def test_higher_accuracy_with_nan_macro_f1_loses_to_valid_candidate():
    nan = float("nan")
    res = {"mean_test_accuracy": [0.95, 0.90], "mean_test_macro_f1": [nan, 0.1],
           "mean_test_neg_log_loss": [-0.1, -0.9]}
    assert select_best_index(res) == 1
    rows = [
        {"model": "a_nan_f1", "cv_accuracy": 0.95, "cv_macro_f1": nan, "cv_log_loss": 0.1},
        {"model": "b_valid", "cv_accuracy": 0.90, "cv_macro_f1": 0.1, "cv_log_loss": 0.9},
    ]
    assert rank_models(rows)["model"].tolist() == ["b_valid", "a_nan_f1"]


def test_all_invalid_candidates_raise():
    nan = float("nan")
    with pytest.raises(ValueError):
        select_best_index({"mean_test_accuracy": [nan, 0.9], "mean_test_macro_f1": [0.5, nan],
                           "mean_test_neg_log_loss": [-0.1, -0.1]})
    with pytest.raises(ValueError):
        rank_models([{"model": "x", "cv_accuracy": nan, "cv_macro_f1": 0.5, "cv_log_loss": 0.1}])


def test_stratified_subsample_keeps_rare_class():
    df = pd.DataFrame({
        TARGET: ["a"] * 100 + ["b"],
        "created_at": pd.date_range("2026-01-01", periods=101, freq="h"),
    })
    out = _stratified_subsample(df, 10)
    assert "b" in set(out[TARGET]) and (out[TARGET] == "a").sum() >= 9
    assert out["created_at"].is_monotonic_increasing and list(out.index) == list(range(len(out)))


def test_artifact_produced_by_cli_loads_with_module_src_models(tmp_path):
    subprocess.run(
        [sys.executable, "-m", "src.train", "--quick", "--models", "merchant_lookup",
         "--out", str(tmp_path / "m")],
        cwd=REPO_ROOT, check=True, capture_output=True,
    )
    assert [p.name for p in tmp_path.iterdir()] == ["m"]
    proc = subprocess.run(
        [sys.executable, "-c",
         "import joblib,sys; b=joblib.load(sys.argv[1]); print(type(b['pipeline'].named_steps['clf']).__module__)",
         str(tmp_path / "m" / "model.joblib")],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "src.models"


def test_gridsearch_refit_callable_agrees_with_best_index(sample_split):
    train, _ = sample_split
    gs = GridSearchCV(
        MODEL_ZOO["logreg"].factory(), {"clf__C": [1.0, 10.0]},
        cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=0),
        scoring={"accuracy": "accuracy", "macro_f1": "f1_macro",
                 "neg_log_loss": "neg_log_loss"},
        refit=select_best_index, n_jobs=1,
    )
    with threadpool_limits(limits=1, user_api="blas"):  # as in run(): avoids BLAS spin-wait thrashing
        gs.fit(train, train[TARGET])
    assert gs.best_index_ == select_best_index(gs.cv_results_)


def test_describe_features_dict(sample_split):
    train, _ = sample_split
    d = _describe_features(MODEL_ZOO["merchant_lookup"].factory().fit(train, train[TARGET]))
    assert set(d) == {"blocks", "input_columns", "text"} and "title_lookup" in d["blocks"]


def test_quick_run_leaves_no_tmp_files(quick_run):
    _, out, _ = quick_run
    assert list(out.glob("*.tmp")) == []
