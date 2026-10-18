import math
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    CAT_COLS,
    DATA_PATH,
    MAJOR_CLASSES,
    MINOR_CLASSES,
    MINOR_TO_MAJOR,
    NUM_COLS,
    TARGET,
    TEXT_NORM_COL,
    add_derived_features,
    build_preprocessor,
    clean,
    dedupe_transactions,
    derive_major,
    frame_from_records,
    load_raw,
    normalise_merchant,
    refund_rule,
    temporal_split,
    title_groups,
)


@pytest.fixture(scope="module")
def full_df():
    return clean(load_raw(DATA_PATH))


@pytest.fixture(scope="module")
def sample_df(full_df):
    return full_df.sample(n=400, random_state=0).reset_index(drop=True)


# ------------------------------------------------------------------ 1. normalisation


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("MAC DONALD S", "mac donald s"),
        ("MC DONALD'S", "mc donald s"),
        ("UBR* PENDING.UBER.COM", "ubr pending uber com"),
        ("INDIGO NE 999990", "indigo ne"),
        ("SumUp  *MASTER POULET", "sumup master poulet"),
        ("Crêperie Élodie", "creperie elodie"),
        ("APPLE.COM/BILL", "apple com bill"),
        ("", ""),
    ],
)
def test_normalise_examples(raw, expected):
    assert normalise_merchant(raw) == expected


def test_missing_values_become_empty():
    assert normalise_merchant(None) == ""
    assert normalise_merchant(float("nan")) == ""
    assert normalise_merchant(math.nan) == ""
    assert normalise_merchant(pd.NA) == ""
    assert normalise_merchant(pd.NaT) == ""


def test_normalise_idempotent():
    s = normalise_merchant("PAYPAL *ITUNESAPPST APPLE")
    assert normalise_merchant(s) == s


# ------------------------------------------------------------------ 2. taxonomy


def test_class_counts():
    assert len(MINOR_CLASSES) == 12
    assert len(MAJOR_CLASSES) == 8


def test_refund_has_no_major():
    assert derive_major("refund") is None


def test_unknown_minor_raises():
    with pytest.raises(KeyError):
        derive_major("unicorns")


def test_mapping_matches_dataset():
    df = pd.read_csv(DATA_PATH, usecols=["minor_category", "major_category"])
    observed = (
        df.dropna(subset=["major_category"])
        .groupby("minor_category")["major_category"]
        .agg(lambda s: set(s))
    )
    for minor, majors in observed.items():
        assert majors == {MINOR_TO_MAJOR[minor]}
    assert set(df["minor_category"].unique()) == set(MINOR_CLASSES)
    assert df.loc[df["minor_category"] == "refund", "major_category"].isna().all()


# ------------------------------------------------------------------ 3. loading / cleaning


def test_load_and_clean_shapes(full_df):
    assert len(full_df) == 8121
    assert full_df["title"].isna().sum() == 0 and (full_df["title"] != "").all()
    assert full_df["created_at"].dt.tz is not None
    for col in CAT_COLS:
        assert full_df[col].isna().sum() == 0
        assert (full_df[col] == full_df[col].str.lower()).all()
    assert set(full_df["pfm_recurring_frequency"]) <= {"none", "monthly"}


def test_clean_title_fallback_to_merchant_name():
    raw = pd.DataFrame({"title": [None, "A"], "merchant_name": ["RETRAIT DAB", None], "amount_value": [-1, -2]})
    out = clean(raw)
    assert out["title"].tolist() == ["RETRAIT DAB", "A"]
    assert out["merchant_name"].tolist() == ["RETRAIT DAB", ""]


@pytest.mark.parametrize("title", [None, "", "   ", np.nan])
def test_blank_title_falls_back_to_merchant(title):
    out = clean(pd.DataFrame({"title": [title], "merchant_name": ["CARREFOUR"], "amount_value": [-1]}))
    assert out["title"].tolist() == ["CARREFOUR"]


def test_omitted_title_falls_back_to_merchant():
    out = clean(pd.DataFrame({"merchant_name": ["CARREFOUR"], "amount_value": [-1]}))
    assert out["title"].tolist() == ["CARREFOUR"]


_TS = [
    "2026-04-01T10:00:00Z",
    "2026-04-01T10:00:00.123Z",
    "2026-04-01 10:00:00",
]


def test_clean_parses_mixed_timestamp_formats_in_any_order():
    aware = datetime(2026, 4, 1, 10, 0, 0, tzinfo=UTC)
    naive = datetime(2026, 4, 1, 10, 0, 0)
    values = [*_TS, aware, naive, None, "garbage"]
    expected = [
        pd.Timestamp("2026-04-01T10:00:00Z"),
        pd.Timestamp("2026-04-01T10:00:00.123Z"),
        pd.Timestamp("2026-04-01T10:00:00Z"),
        pd.Timestamp("2026-04-01T10:00:00Z"),
        pd.Timestamp("2026-04-01T10:00:00Z"),
        pd.NaT,
        pd.NaT,
    ]
    for order in (list(range(7)), list(range(6, -1, -1)), [3, 0, 5, 1, 6, 2, 4]):
        raw = pd.DataFrame(
            {"title": ["x"] * 7, "amount_value": [-1.0] * 7, "created_at": pd.Series([values[i] for i in order], dtype=object)}
        )
        got = clean(raw)["created_at"].tolist()
        for g, i in zip(got, order, strict=False):
            e = expected[i]
            assert (pd.isna(g) and pd.isna(e)) or g == e, (order, i, g, e)


def test_digit_only_title_is_kept_and_groups_to_empty_string():
    out = clean(pd.DataFrame({"title": ["999990"], "amount_value": [-20.0]}))
    assert len(out) == 1 and out["title"].tolist() == ["999990"]
    assert normalise_merchant("999990") == ""
    assert title_groups(out).tolist() == [""]


def test_lowercase_and_offset_timestamps():
    forms = ["2026-04-01t10:00:00z", "2026-04-01T10:00:00Z", "2026-04-01T10:00:00.000+00:00"]
    frame = clean(pd.DataFrame({"title": ["x"] * 3, "amount_value": [-1.0] * 3, "created_at": forms}))
    assert frame["created_at"].notna().all()
    assert (frame["created_at"] == pd.Timestamp("2026-04-01T10:00:00Z")).all()
    feats = add_derived_features(frame)
    assert (feats["hour"] == 10).all() and (feats["dow"] == 2).all()


# ------------------------------------------------------------------ 4. frame_from_records


def _parity_cols():
    return CAT_COLS + NUM_COLS + [TEXT_NORM_COL]


def test_records_feature_parity_with_clean():
    raw = load_raw(DATA_PATH).sample(n=200, random_state=1).reset_index(drop=True)
    expected = add_derived_features(clean(raw))
    got = add_derived_features(frame_from_records(raw.to_dict("records")))
    pd.testing.assert_frame_equal(got[_parity_cols()], expected[_parity_cols()])


def test_minimal_record_defaults():
    got = frame_from_records([{"title": "NETFLIX", "amount_value": -9.0}])
    assert got.loc[0, "status"] == "completed"
    assert got.loc[0, "pfm_recurring_frequency"] == "none"
    expected = clean(pd.DataFrame({"title": ["NETFLIX"], "amount_value": [-9.0]}))
    pd.testing.assert_frame_equal(
        add_derived_features(got)[_parity_cols()], add_derived_features(expected)[_parity_cols()]
    )


def test_records_ignore_unknown_keys():
    base = {"title": "NETFLIX", "amount_value": -9.0}
    plain = frame_from_records([base])
    extra = frame_from_records([{**base, "bogus": "zzz", "ref_id": "r1"}])
    pd.testing.assert_frame_equal(plain, extra)
    assert "bogus" not in extra.columns


def test_records_merchant_name_fallback_and_missing_created_at():
    df = frame_from_records(
        [
            {"merchant_name": "LA BANQUE POSTALE", "amount_value": -50, "fees": 100},
            {"title": "NETFLIX.COM", "amount_value": -13.49, "created_at": "2026-04-01T10:00:00Z"},
        ]
    )
    assert df["title"].tolist() == ["LA BANQUE POSTALE", "NETFLIX.COM"]
    assert df["created_at"].isna().tolist() == [True, False]


# ------------------------------------------------------------------ 5. dedupe


def test_dedupe_removes_exact_duplicates_except_transaction_id(full_df):
    out = dedupe_transactions(full_df)
    assert len(full_df) == 8121 and len(out) == 8001
    assert out["ref_id"].is_unique
    assert out.index.equals(pd.RangeIndex(len(out)))
    assert len(dedupe_transactions(out)) == len(out)
    assert out["transaction_id"].iloc[0] == full_df["transaction_id"].iloc[0]


def test_dedupe_without_id_column():
    df = clean(pd.DataFrame({"title": ["A", "A", "B"], "amount_value": [-1.0, -1.0, -2.0]}))
    assert len(dedupe_transactions(df)) == 2


def test_dedupe_keeps_first_occurrence_ids_and_order():
    df = clean(pd.DataFrame({
        "transaction_id": ["t1", "t2", "t3", "t4", "t5"],
        "title": ["A", "B", "A", "B", "C"],
        "amount_value": [-1.0, -2.0, -1.0, -2.0, -3.0],
    }))
    out = dedupe_transactions(df)
    assert out["transaction_id"].tolist() == ["t1", "t2", "t5"]
    assert out["title"].tolist() == ["A", "B", "C"]
    assert dedupe_transactions(out)["transaction_id"].tolist() == ["t1", "t2", "t5"]


# ------------------------------------------------------------------ 6. temporal split


def test_temporal_split_is_ordered_and_disjoint(full_df):
    train, test = temporal_split(full_df, 0.8)
    assert len(train) + len(test) == len(full_df)
    assert train["created_at"].max() < test["created_at"].min()
    assert abs(len(train) / len(full_df) - 0.8) < 0.01
    assert set(test[TARGET]) <= set(train[TARGET])


def test_temporal_split_rejects_bad_inputs():
    df = clean(pd.DataFrame({
        "title": list("abcd"), "amount_value": [-1.0] * 4,
        "created_at": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
    }))
    with pytest.raises(ValueError):
        temporal_split(df.iloc[0:0])
    for frac in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            temporal_split(df, frac)
    with_nat = df.copy()
    with_nat["created_at"] = pd.to_datetime(["2026-01-01", None, "2026-01-03", "2026-01-04"], utc=True)
    with pytest.raises(ValueError):
        temporal_split(with_nat)


def test_temporal_split_ties_never_straddle():
    df = clean(pd.DataFrame({
        "title": list("abcdef"), "amount_value": [-1.0] * 6,
        "created_at": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-03", "2026-01-03", "2026-01-04"],
    }))
    train, test = temporal_split(df, 0.5)
    assert train["created_at"].max() < test["created_at"].min()
    assert len(train) + len(test) == 6
    with pytest.raises(ValueError):
        temporal_split(df.assign(created_at=df["created_at"].iloc[0]), 0.5)


# ------------------------------------------------------------------ 7. derived features


def test_derived_columns_present_and_finite(sample_df):
    df = add_derived_features(sample_df)
    assert TEXT_NORM_COL in df.columns
    for col in NUM_COLS:
        assert np.isfinite(df[col]).all(), col
    assert set(df["amount_sign"].unique()) <= {-1.0, 0.0, 1.0}
    assert df["hour"].between(0, 23).all()


def test_derived_features_missing_created_at():
    df = clean(pd.DataFrame({"title": ["X"], "amount_value": [-3.0]}))
    out = add_derived_features(df)
    assert out.loc[0, "hour"] == -1 and out.loc[0, "dow"] == -1 and out.loc[0, "is_weekend"] == 0


def test_round_amount_indicator():
    df = clean(pd.DataFrame({"title": list("abcde"), "amount_value": [-40.0, -40.01, 0.0, 100.0, -7.5]}))
    assert add_derived_features(df)["is_round_amount"].tolist() == [1.0, 0.0, 0.0, 1.0, 0.0]


def test_round_amount_tolerance_is_absolute_and_symmetric():
    amounts = [-39.99999, -40.0000000001, -40.00001, -9.99999, -10.0000000001]
    df = clean(pd.DataFrame({"title": list("abcde"), "amount_value": amounts}))
    assert add_derived_features(df)["is_round_amount"].tolist() == [0.0, 1.0, 0.0, 0.0, 1.0]


# ------------------------------------------------------------------ 8. preprocessor


def test_preprocessor_fit_transform_shapes_and_determinism(sample_df):
    df = add_derived_features(sample_df)
    x1 = build_preprocessor().fit_transform(df)
    x2 = build_preprocessor().fit_transform(df)
    assert x1.shape[0] == len(df) and x1.shape[1] > 100
    assert abs(x1 - x2).sum() == 0


def test_preprocessor_handles_unknown_category(sample_df):
    df = add_derived_features(sample_df)
    pre = build_preprocessor().fit(df)
    new = df.head(1).copy()
    new["digitalization_type"] = "klarna"
    assert pre.transform(new).shape[0] == 1


def test_card_name_not_a_model_feature(sample_df):
    assert "card_name" not in CAT_COLS
    pre = build_preprocessor().fit(add_derived_features(sample_df))
    assert not any("card_name" in name for name in pre.get_feature_names_out())


# ------------------------------------------------------------------ 9. refund_rule


def test_refund_rule_equals_label_on_full_data(full_df):
    rule = refund_rule(full_df)
    label = (full_df[TARGET] == "refund").to_numpy()
    assert rule.dtype == bool
    assert rule.sum() == 123
    np.testing.assert_array_equal(rule, label)


def test_refund_rule_hand_built():
    df = pd.DataFrame({
        "status": ["Refunded ", "refunded", "completed", "Completed", None],
        "amount_value": [-5.0, -5.0, -5.0, 12.0, -1.0],
    })
    np.testing.assert_array_equal(refund_rule(df), [True, True, False, True, False])


def test_refund_rule_without_status_column():
    df = pd.DataFrame({"amount_value": [-5.0, 0.0, 3.0]})
    np.testing.assert_array_equal(refund_rule(df), [False, False, True])


def test_refund_rule_categorical_status_with_missing():
    df = pd.DataFrame({
        "status": pd.Series(["refunded", None, "completed"], dtype="category"),
        "amount_value": [-5.0, -5.0, -5.0],
    })
    np.testing.assert_array_equal(refund_rule(df), [True, False, False])


def test_refund_rule_nullable_boolean_status():
    df = pd.DataFrame({
        "status": pd.array([True, None], dtype="boolean"),
        "amount_value": [-1.0, 2.0],
    })
    np.testing.assert_array_equal(refund_rule(df), [False, True])


def test_refund_rule_without_amount_column():
    df = pd.DataFrame({"status": ["refunded", "completed", None]})
    np.testing.assert_array_equal(refund_rule(df), [True, False, False])


# ------------------------------------------------------------------ 10. title_groups


def test_title_groups_match_normalised_text(sample_df):
    expected = add_derived_features(sample_df)[TEXT_NORM_COL].to_numpy()
    np.testing.assert_array_equal(title_groups(sample_df), expected)


def test_title_groups_empty_titles_map_to_empty_string():
    df = clean(pd.DataFrame({"title": ["", "999990", "CARREFOUR 12"], "amount_value": [-1.0] * 3}))
    assert title_groups(df).tolist() == ["", "", "carrefour"]
