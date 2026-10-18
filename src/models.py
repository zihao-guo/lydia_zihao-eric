"""Model zoo: every entry is a factory returning an sklearn Pipeline that consumes the cleaned
DataFrame (see src.preprocessing.clean) and predicts `minor_category`.

The estimator classes live in this importable module on purpose. If they were defined in
`src/train.py`, a model trained with `python -m src.train` would be pickled as
`__main__.MerchantLookupClassifier` and could not be loaded by any other process (the API, the
evaluation CLI, a fresh interpreter).

The zoo holds five pipelines: two baselines (majority prior, merchant lookup), two linear models
(logistic regression, calibrated LinearSVC) and one text-only ablation of the calibrated SVC.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

from src.preprocessing import SEED, TEXT_NORM_COL, add_derived_features, build_preprocessor


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    description: str
    factory: Callable[[], Pipeline]
    param_grid: dict[str, list] | None = None


def make_pipeline(preprocessor, clf) -> Pipeline:
    """derive -> features -> clf. `preprocessor` may be None (clf consumes the derived frame)."""
    steps = [("derive", FunctionTransformer(add_derived_features))]
    if preprocessor is not None:
        steps.append(("features", preprocessor))
    steps.append(("clf", clf))
    return Pipeline(steps)


class MerchantLookupClassifier(ClassifierMixin, BaseEstimator):
    """Baseline: Laplace-smoothed label distribution per normalised merchant string;
    unseen merchants get the training prior."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    @staticmethod
    def _keys(X) -> list[str]:
        if isinstance(X, pd.DataFrame):
            return X[TEXT_NORM_COL].astype(str).tolist()
        return [str(v) for v in np.asarray(X).ravel()]

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        index = {c: i for i, c in enumerate(self.classes_)}
        counts: dict[str, np.ndarray] = {}
        for key, label in zip(self._keys(X), y, strict=False):
            counts.setdefault(key, np.zeros(len(self.classes_)))[index[label]] += 1
        k = len(self.classes_)
        self.table_ = {key: (c + self.alpha) / (c.sum() + self.alpha * k) for key, c in counts.items()}
        prior = np.bincount([index[v] for v in y], minlength=k).astype(float)
        self.prior_ = prior / prior.sum()
        return self

    def predict_proba(self, X):
        return np.vstack([self.table_.get(key, self.prior_) for key in self._keys(X)])

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def _majority() -> Pipeline:
    return make_pipeline(None, DummyClassifier(strategy="prior"))


def _merchant_lookup() -> Pipeline:
    return make_pipeline(None, MerchantLookupClassifier(alpha=1.0))


def _logreg() -> Pipeline:
    return make_pipeline(build_preprocessor(), LogisticRegression(C=10.0, max_iter=3000))


def _calibrated_svc() -> CalibratedClassifierCV:
    return CalibratedClassifierCV(LinearSVC(C=0.5, max_iter=20000, random_state=SEED), cv=3, method="sigmoid")


def _linear_svc() -> Pipeline:
    return make_pipeline(build_preprocessor(), _calibrated_svc())


def _linear_svc_text() -> Pipeline:
    return make_pipeline(build_preprocessor(cats=False, nums=False), _calibrated_svc())


MODEL_ZOO: dict[str, ModelSpec] = {
    "majority": ModelSpec(
        "majority", "baseline", "Always predicts the training class prior (no features).",
        _majority,
    ),
    "merchant_lookup": ModelSpec(
        "merchant_lookup", "baseline",
        "Laplace-smoothed label distribution per normalised merchant string; prior for unseen merchants.",
        _merchant_lookup,
    ),
    "logreg": ModelSpec(
        "logreg", "linear", "TF-IDF + categorical + numeric features, multinomial logistic regression.",
        _logreg, {"clf__C": [1.0, 10.0, 30.0], "clf__class_weight": [None, "balanced"]},
    ),
    "linear_svc": ModelSpec(
        "linear_svc", "linear", "TF-IDF + categorical + numeric features, sigmoid-calibrated LinearSVC.",
        _linear_svc, {"clf__estimator__C": [0.1, 0.5, 1.0]},
    ),
    "linear_svc_text": ModelSpec(
        "linear_svc_text", "ablation",
        "Calibrated LinearSVC on normalised-title TF-IDF only (no categoricals/numerics) "
        "- isolates the contribution of the structured features.",
        _linear_svc_text, {"clf__estimator__C": [0.1, 0.5, 1.0]},
    ),
}
