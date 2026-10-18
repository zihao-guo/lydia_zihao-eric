"""Pydantic request/response models for the prediction API."""
from __future__ import annotations

import math
import re
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.predict import coerce_amount
from src.preprocessing import DEFAULT_STATUS


class TransactionIn(BaseModel):
    """One card transaction as known at transaction time (targets excluded).

    Identifier / PII columns from the CSV (ref_id, member_id, member_name, merchant_id, card_type)
    are accepted but ignored.
    """

    model_config = ConfigDict(extra="ignore", coerce_numbers_to_str=True, str_strip_whitespace=True)

    transaction_id: str | None = None
    created_at: datetime | None = Field(
        None, description="ISO 8601 timestamp (a date-only value means midnight UTC); numeric timestamps are rejected"
    )
    merchant_name: str | None = None
    title: str | None = None
    amount_value: float = Field(
        ...,
        allow_inf_nan=False,
        ge=-1e9,
        le=1e9,
        description="Signed amount; negative = debit, positive = credit/refund",
    )
    amount_currency: str = "EUR"
    fees: int = Field(0, ge=0, le=100_000_000, description="Fees in cents, 0-100,000,000")
    card_name: str | None = None
    digitalization_type: str | None = None
    merchant_pan_entry_method: str | None = None
    pfm_recurring_frequency: str | None = None
    status: str = DEFAULT_STATUS

    @field_validator("amount_value", mode="before")
    @classmethod
    def _amount(cls, v):
        return coerce_amount(v)

    @field_validator("fees", mode="before")
    @classmethod
    def _fees(cls, v):
        if isinstance(v, bool):
            raise ValueError("fees must be an integer number of cents, not a boolean")
        return v

    @field_validator("title", "merchant_name", mode="before")
    @classmethod
    def _text_only_str(cls, v):
        """Non-string text is treated as missing, exactly like the predictor does."""
        return v if isinstance(v, str) else None

    @field_validator("created_at", mode="before")
    @classmethod
    def _created_at(cls, v):
        if isinstance(v, str):
            if re.fullmatch(r"\s*[-+]?\d+(\.\d+)?\s*", v):
                raise ValueError("created_at must be an ISO 8601 string")
            return v
        if v is None or isinstance(v, datetime):
            return v
        raise ValueError("created_at must be an ISO 8601 string")

    @model_validator(mode="before")
    @classmethod
    def _drop_nulls(cls, data):
        """Treat float NaN / None like a missing value (CSV records carry NaN), so that
        coerce_numbers_to_str never turns NaN into the string "nan". Defaults then apply
        to non-optional fields; amount_value is left alone so NaN is rejected."""
        if not isinstance(data, dict):
            return data
        out = {}
        for k, v in data.items():
            if k != "amount_value" and isinstance(v, float) and math.isnan(v):
                v = None
            if v is None and k in ("status", "amount_currency", "fees"):
                continue
            out[k] = v
        return out

    @model_validator(mode="after")
    def _text_required(self) -> TransactionIn:
        if not ((self.title or "").strip() or (self.merchant_name or "").strip()):
            raise ValueError("needs a non-blank title or merchant_name")
        return self


class ClassProbability(BaseModel):
    label: str
    probability: float


class PredictionOut(BaseModel):
    transaction_id: str | None = None
    minor_category: str
    major_category: str | None
    confidence: float
    top_k: list[ClassProbability]
    model_version: str


class HealthOut(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None
