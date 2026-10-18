import datetime as dt

import joblib
import pytest


@pytest.fixture(scope="session")
def tiny_bundle_path(tmp_path_factory):
    """Small merchant_lookup bundle ({"pipeline", "metadata"}) for predictor tests.

    src.models is imported inside the function so that modules which do not need it
    (e.g. test_preprocessing.py) can be collected before src/models.py exists.
    """
    from src.models import MODEL_ZOO
    from src.preprocessing import DATA_PATH, TARGET, clean, load_raw

    df = clean(load_raw(DATA_PATH)).sample(n=1000, random_state=0)
    pipe = MODEL_ZOO["merchant_lookup"].factory().fit(df, df[TARGET])
    path = tmp_path_factory.mktemp("bundle") / "model.joblib"
    joblib.dump(
        {
            "pipeline": pipe,
            "metadata": {
                "model_name": "merchant_lookup",
                "model_version": "test-0",
                "trained_at": dt.datetime(2026, 1, 1, tzinfo=dt.UTC).isoformat(),
                "classes": list(pipe.classes_),
            },
        },
        path,
    )
    return path
