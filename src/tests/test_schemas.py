import pytest
from pydantic import ValidationError

from api.schemas import PredictionOut, TransactionIn


def test_minimal_valid_request():
    t = TransactionIn(title="CARREFOUR CITY", amount_value=-12.3)
    assert t.status == "completed" and t.fees == 0 and t.amount_currency == "EUR"


def test_requires_title_or_merchant_name():
    with pytest.raises(ValidationError):
        TransactionIn(amount_value=-1.0)
    with pytest.raises(ValidationError):
        TransactionIn(title="   ", merchant_name="", amount_value=-1.0)


def test_merchant_name_alone_is_enough():
    t = TransactionIn(merchant_name="LA BANQUE POSTALE", amount_value=-50)
    assert t.title is None


def test_transaction_id_number_coerced():
    t = TransactionIn(transaction_id=10022027, title="X", amount_value=-1)
    assert t.transaction_id == "10022027"


def test_extra_columns_ignored_and_dates_parsed():
    t = TransactionIn(
        title="TOTAL", amount_value=-53.26, member_id=1, ref_id="abc",
        created_at="2026-03-28T12:51:29Z",
    )
    assert t.created_at.year == 2026 and t.created_at.tzinfo is not None
    assert not hasattr(t, "member_id")


def test_prediction_out_roundtrip():
    p = PredictionOut(
        transaction_id="1", minor_category="refund", major_category=None, confidence=1.0,
        top_k=[{"label": "refund", "probability": 1.0}], model_version="v",
    )
    assert p.model_dump()["major_category"] is None


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), "NaN", "Infinity", "-Infinity", "inf"])
def test_nonfinite_amount_rejected(bad):
    with pytest.raises(ValidationError):
        TransactionIn(title="X", amount_value=bad)


def test_nan_optional_fields_become_missing_not_nan_string():
    nan = float("nan")
    t = TransactionIn(
        title="X", amount_value=-1, pfm_recurring_frequency=nan, card_name=nan, status=nan,
        merchant_name=nan, created_at=nan, fees=nan,
    )
    assert t.pfm_recurring_frequency is None and t.card_name is None and t.merchant_name is None
    assert t.status == "completed" and t.fees == 0 and t.created_at is None


@pytest.mark.parametrize("extra", [{"fees": -1}, {"fees": 100_000_001}, {"fees": 9223372036854775808},
                                   {"amount_value": 1e10}, {"amount_value": -1e10}])
def test_out_of_range_values_rejected(extra):
    with pytest.raises(ValidationError):
        TransactionIn(**{"title": "X", "amount_value": -1.0, **extra})


def test_range_boundaries_accepted():
    assert TransactionIn(title="X", amount_value=-1e9, fees=100_000_000).fees == 100_000_000


@pytest.mark.parametrize("bad", [True, False, "1_0", "abc", "", "1e999", "0x10", None, [1]])
def test_amount_strictness(bad):
    with pytest.raises(ValidationError):
        TransactionIn(title="X", amount_value=bad)


def test_amount_numeric_string_accepted_as_float():
    t = TransactionIn(title="X", amount_value=" -12.5 ")
    assert t.amount_value == -12.5 and isinstance(t.amount_value, float)


def test_fees_bool_rejected():
    with pytest.raises(ValidationError):
        TransactionIn(title="X", amount_value=-1, fees=True)


def test_non_string_text_is_missing():
    t = TransactionIn(title=False, merchant_name="CARREFOUR CITY", amount_value=-1)
    assert t.title is None and t.merchant_name == "CARREFOUR CITY"
    with pytest.raises(ValidationError, match="non-blank title or merchant_name"):
        TransactionIn(title=123, amount_value=-1)


def test_created_at_iso_only():
    assert TransactionIn(title="X", amount_value=-1, created_at="2026-03-16T19:51:25Z").created_at.year == 2026
    for bad in (1777376721, 1777376721.5, True):
        with pytest.raises(ValidationError):
            TransactionIn(title="X", amount_value=-1, created_at=bad)


@pytest.mark.parametrize("bad", ["1777376721", " 1777376721.5 ", "-5", "+7"])
def test_created_at_numeric_string_rejected(bad):
    with pytest.raises(ValidationError, match="ISO 8601"):
        TransactionIn(title="X", amount_value=-1, created_at=bad)


def test_created_at_date_only_accepted():
    assert TransactionIn(title="X", amount_value=-1, created_at="2026-03-16").created_at.year == 2026
