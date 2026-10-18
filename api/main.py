"""FastAPI prediction service.

Start: `uvicorn api.main:app --port 8000`; web form at `/`, OpenAPI docs at `/docs`.

Model loading policy: a MISSING artifact degrades gracefully (warning logged, model endpoints
answer 503, /health reports unavailable). An artifact that exists but cannot be loaded (corrupt
file, directory, missing refund class) is logged and re-raised so the service fails fast at startup.
Request bodies are never logged or echoed in error responses.
"""
from __future__ import annotations

import logging
import math
import os
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse

from api.schemas import HealthOut, PredictionOut, TransactionIn
from src import __version__
from src.predict import Predictor, load_bundle
from src.preprocessing import MODEL_PATH

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

logger = logging.getLogger(__name__)
MAX_BATCH = 1000


def model_path() -> Path:
    return Path(os.environ.get("LYDIA_MODEL_PATH", str(MODEL_PATH)))


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = None
    path = model_path()
    if path.exists():
        try:
            app.state.predictor = Predictor(load_bundle(path))
        except Exception:
            logger.exception("could not load model from %s", path)
            raise
        logger.info("loaded model %s from %s", app.state.predictor.version, path)
    else:
        logger.warning("model artifact not found at %s; run `python -m src.train`", path)
    yield


app = FastAPI(
    title="Transaction Labeling API",
    version=__version__,
    description="Predicts the minor and major category of a single card transaction.",
    lifespan=lifespan,
)


@lru_cache(maxsize=1)
def _index_html() -> str:
    return (Path(__file__).with_name("index.html")).read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> HTMLResponse:
    """Web form for one transaction; the JSON API is documented at /docs."""
    return HTMLResponse(_index_html())


def _sanitise(obj):
    """Replace non-finite floats (not JSON serialisable) by their string repr, recursively."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return repr(obj)
    if isinstance(obj, dict):
        return {k: _sanitise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitise(v) for v in obj]
    return obj


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = []
    for e in exc.errors():
        entry = {k: e[k] for k in ("type", "loc", "msg") if k in e}
        ctx = e.get("ctx")
        if isinstance(ctx, dict) and all(v is None or isinstance(v, (str, int, float, bool)) for v in ctx.values()):
            entry["ctx"] = _sanitise(ctx)
        errors.append(entry)
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(errors)})


def _unprocessable(exc: ValueError) -> HTTPException:
    """Predictor ValueError -> 422 with the same list-shaped detail pydantic produces."""
    return HTTPException(
        status_code=422, detail=[{"loc": ["body"], "msg": str(exc), "type": "value_error"}]
    )


def _predictor(request: Request) -> Predictor:
    predictor = request.app.state.predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return predictor


@app.get("/health", response_model=HealthOut)
def health(request: Request):
    predictor = request.app.state.predictor
    if predictor is None:
        out = HealthOut(status="unavailable", model_loaded=False, model_version=None)
        return JSONResponse(status_code=503, content=out.model_dump())
    return HealthOut(status="ok", model_loaded=True, model_version=predictor.version)


@app.get("/v1/model")
def model_info(request: Request) -> dict:
    return _predictor(request).metadata


@app.post("/v1/predict", response_model=PredictionOut)
def predict(request: Request, txn: TransactionIn) -> PredictionOut:
    predictor = _predictor(request)
    try:
        return predictor.predict_one(txn.model_dump())
    except ValueError as exc:
        raise _unprocessable(exc) from exc


@app.post("/v1/predict/batch", response_model=list[PredictionOut])
def predict_batch(request: Request, txns: list[TransactionIn] = Body(..., max_length=MAX_BATCH)) -> list[PredictionOut]:
    predictor = _predictor(request)
    try:
        return predictor.predict([t.model_dump() for t in txns])
    except ValueError as exc:
        raise _unprocessable(exc) from exc
