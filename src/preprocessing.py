"""Single shared preprocessing module.

Used by training, evaluation, serving and the EDA notebook so that every stage sees
exactly the same constants, text normalisation, taxonomy, cleaning, splitting and
feature engineering. The persisted model pickles ``add_derived_features`` from here,
so the import path ``src.preprocessing`` must stay stable.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------- constants

SEED = 42

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "transactions.csv"
MODELS_DIR = REPO_ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"

# Temporal split: first TRAIN_FRACTION of rows ordered by created_at is training.
TRAIN_FRACTION = 0.8

# Value assumed when a transaction carries no status (training and serving).
DEFAULT_STATUS = "completed"

TARGET = "minor_category"
MAJOR_TARGET = "major_category"

# Text: `title` is never null in the data and equals merchant_name when present.
TEXT_COL = "title"
TEXT_NORM_COL = "text_norm"

CAT_COLS = [
    "digitalization_type",
    "merchant_pan_entry_method",
    "pfm_recurring_frequency",
    "status",
]

# Derived by add_derived_features
NUM_COLS = ["amount_abs_log", "amount_sign", "fees_log", "has_fees", "hour", "dow", "is_weekend", "is_round_amount"]

# Columns the model consumes from a raw transaction
RAW_INPUT_COLS = [
    "transaction_id",
    "created_at",
    "merchant_name",
    "title",
    "amount_value",
    "fees",
    "card_name",
    "digitalization_type",
    "merchant_pan_entry_method",
    "pfm_recurring_frequency",
    "status",
]

# Columns ignored when detecting exact duplicate transactions
DEDUPE_IGNORE_COLS = ["transaction_id"]

# Deliberately excluded from features (identifiers / PII / leakage risk)
ID_COLS = ["transaction_id", "ref_id", "member_id", "member_name", "merchant_id"]
# Constant in the dataset
CONSTANT_COLS = ["amount_currency", "card_type"]

# --------------------------------------------------------------------------- taxonomy
# Minor -> major category. Verified against the dataset: the mapping is a strict
# function and `refund` rows have no major category.

MINOR_TO_MAJOR: dict[str, str | None] = {
    "atm": "finance",
    "betting": "entertainment",
    "hairdresser": "self_care",
    "multimedia": "entertainment",
    "others_shopping": "shopping",
    "pharmacies": "self_care",
    "refund": None,
    "restaurants": "restaurants_bars_cafes",
    "supermarkets": "groceries",
    "tolls_gas_stations": "transportation",
    "transport_means": "transportation",
    "travel_means": "travel",
}

MINOR_CLASSES: list[str] = sorted(MINOR_TO_MAJOR)
MAJOR_CLASSES: list[str] = sorted({m for m in MINOR_TO_MAJOR.values() if m is not None})


def derive_major(minor: str) -> str | None:
    """Return the major category for a minor one (None for refund). Raises KeyError if unknown."""
    if minor not in MINOR_TO_MAJOR:
        raise KeyError(f"unknown minor_category: {minor!r}")
    return MINOR_TO_MAJOR[minor]


# --------------------------------------------------------------------------- text

_DIGITS = re.compile(r"\d+")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalise_merchant(text: object) -> str:
    """Lower-case, strip accents, drop digits and punctuation, collapse whitespace.

    Examples: "UBR* PENDING.UBER.COM" -> "ubr pending uber com"; "INDIGO NE 999990" -> "indigo ne".
    None / NaN / pd.NA / NaT -> "".
    """
    try:
        missing = bool(pd.isna(text))
    except (TypeError, ValueError):  # non-scalar (list, array...): not a missing scalar
        missing = False
    if missing:
        return ""
    s = unicodedata.normalize("NFKD", str(text))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = _DIGITS.sub(" ", s)
    s = _NON_ALNUM.sub(" ", s)
    return " ".join(s.split())


# --------------------------------------------------------------------------- data

_CAT_DEFAULTS = {col: "unknown" for col in CAT_COLS}
_CAT_DEFAULTS["pfm_recurring_frequency"] = "none"
_CAT_DEFAULTS["status"] = DEFAULT_STATUS


def load_raw(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Read the CSV keeping identifier columns as strings."""
    return pd.read_csv(
        path, dtype={"transaction_id": str, "ref_id": str, "member_id": str, "merchant_id": str}
    )


def _blank_to_na(col: pd.Series) -> pd.Series:
    s = col.astype("string").str.strip()
    return s.mask(s == "", pd.NA)


def parse_utc(col: pd.Series) -> pd.Series:
    """Parse timestamps of mixed formats (with/without fractions, lowercase t/z, tz-aware
    datetimes) to UTC; unparseable values become NaT."""
    return pd.to_datetime(col, utc=True, errors="coerce", format="mixed")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same cleaning to training rows and API rows.

    - guarantees every RAW_INPUT_COLS column exists
    - `title` falls back to `merchant_name`; both end up as non-null str
    - categoricals: filled (`unknown`, `none` for recurrence), stripped, lower-cased
    - `created_at` -> tz-aware UTC (NaT when missing/invalid); `amount_value` float; `fees` int
    """
    out = df.copy()
    for col in RAW_INPUT_COLS:
        if col not in out.columns:
            out[col] = None

    # Blank / whitespace-only strings count as missing so the fallback applies.
    title = _blank_to_na(out["title"])
    merchant = _blank_to_na(out["merchant_name"])
    out["title"] = title.fillna(merchant).fillna("").astype(str)
    out["merchant_name"] = merchant.fillna("").astype(str)

    for col, default in _CAT_DEFAULTS.items():
        out[col] = out[col].astype("string").fillna(default).str.strip().str.lower().astype(str)

    out["created_at"] = parse_utc(out["created_at"])
    out["amount_value"] = pd.to_numeric(out["amount_value"], errors="coerce").fillna(0.0).astype(float)
    out["fees"] = pd.to_numeric(out["fees"], errors="coerce").fillna(0).astype(int)
    return out


def dedupe_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate transactions (every column equal except DEDUPE_IGNORE_COLS), keep first."""
    subset = [c for c in df.columns if c not in DEDUPE_IGNORE_COLS]
    return df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)


def temporal_split(
    df: pd.DataFrame, train_fraction: float = TRAIN_FRACTION
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Order by created_at; earliest `train_fraction` is training, the rest is test."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be strictly between 0 and 1, got {train_fraction}")
    if len(df) == 0:
        raise ValueError("cannot split an empty frame")
    if df["created_at"].isna().any():
        raise ValueError("created_at contains missing values; cannot split temporally")
    ordered = df.sort_values("created_at", kind="mergesort").reset_index(drop=True)
    cut_time = ordered["created_at"].iloc[int(len(ordered) * train_fraction)]
    train = ordered[ordered["created_at"] < cut_time].reset_index(drop=True)
    test = ordered[ordered["created_at"] >= cut_time].reset_index(drop=True)
    if train.empty or test.empty:
        raise ValueError("temporal split produced an empty train or test set (too few distinct timestamps)")
    return train, test


def frame_from_records(records: list[dict]) -> pd.DataFrame:
    """Build a cleaned frame (same columns/cleaning as training) from raw dict records.

    Unknown keys are dropped; missing keys become None before cleaning. This is not a
    validator (bool amounts become 1.0, invalid/missing amounts become 0.0); callers that
    accept external input must validate before cleaning.
    """
    df = pd.DataFrame(list(records), columns=RAW_INPUT_COLS)
    return clean(df)


# --------------------------------------------------------------------------- features


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add `text_norm` and the numeric columns listed in NUM_COLS.

    Expects a frame produced by `clean()` (columns amount_value, fees, created_at present).
    """
    out = df.copy()
    out[TEXT_NORM_COL] = out[TEXT_COL].map(normalise_merchant)

    amount = pd.to_numeric(out["amount_value"], errors="coerce").fillna(0.0).astype(float)
    out["amount_abs_log"] = np.log1p(amount.abs())
    out["amount_sign"] = np.sign(amount).astype(float)
    abs_amount = amount.abs()
    out["is_round_amount"] = (
        (abs_amount > 0)
        & (np.isclose(abs_amount % 10, 0, rtol=0, atol=1e-9)
            | np.isclose(abs_amount % 10, 10, rtol=0, atol=1e-9))
    ).astype(float)

    fees = pd.to_numeric(out["fees"], errors="coerce").fillna(0.0).astype(float)
    out["fees_log"] = np.log1p(fees.clip(lower=0))
    out["has_fees"] = (fees > 0).astype(float)

    ts = parse_utc(out["created_at"])
    dow = ts.dt.dayofweek
    out["hour"] = ts.dt.hour.fillna(-1).astype(float)
    out["dow"] = dow.fillna(-1).astype(float)
    out["is_weekend"] = (dow >= 5).fillna(False).astype(float)
    return out


def build_preprocessor(
    text: bool = True,
    cats: bool = True,
    nums: bool = True,
    word: bool = True,
    min_df: int = 2,
) -> ColumnTransformer:
    """Sparse feature block: TF-IDF on normalised text + one-hot categoricals + scaled numerics."""
    transformers: list = []
    if text:
        transformers.append(
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(2, 5), min_df=min_df,
                    sublinear_tf=True, dtype=np.float32,
                ),
                TEXT_NORM_COL,
            )
        )
        if word:
            transformers.append(
                (
                    "word",
                    TfidfVectorizer(
                        analyzer="word", ngram_range=(1, 2), min_df=min_df, sublinear_tf=True,
                        token_pattern=r"(?u)\b\w+\b", dtype=np.float32,
                    ),
                    TEXT_NORM_COL,
                )
            )
    if cats:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), CAT_COLS))
    if nums:
        transformers.append(("num", StandardScaler(with_mean=False), NUM_COLS))
    return ColumnTransformer(transformers, sparse_threshold=0.3)


# --------------------------------------------------------------------------- rules / groups


def refund_rule(df: pd.DataFrame) -> np.ndarray:
    """Deterministic refund guard: status is 'refunded' or the amount is positive (boolean array)."""
    if "status" in df.columns:
        status = df["status"].astype("string").fillna(DEFAULT_STATUS)
    else:
        status = pd.Series(DEFAULT_STATUS, index=df.index)
    refunded = status.astype(str).str.strip().str.lower() == "refunded"
    amount = df["amount_value"] if "amount_value" in df.columns else pd.Series(0.0, index=df.index)
    positive = pd.to_numeric(amount, errors="coerce").fillna(0) > 0
    return (refunded | positive).to_numpy(dtype=bool)


def title_groups(df: pd.DataFrame) -> np.ndarray:
    """Normalised titles, used as the `groups` argument of a title-grouped CV.

    Expects a frame produced by `clean()` (columns amount_value, fees, created_at present).
    """
    return add_derived_features(df)[TEXT_NORM_COL].to_numpy()
