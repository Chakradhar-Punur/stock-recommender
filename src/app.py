"""
Run with (from the project root):
    cd src && ../venv/bin/uvicorn app:app --reload --port 8000
"""

import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from inference import predict, load_model, FEATURE_COLUMNS
from explain import explain

CACHE_TTL_SECONDS = 60 * 60

app = FastAPI(title="Stock Recommender API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    _model = load_model()
    _feature_importance = dict(
        sorted(
            zip(FEATURE_COLUMNS, (float(v) for v in _model.feature_importances_)),
            key=lambda kv: kv[1],
            reverse=True,
        )
    )
except FileNotFoundError as e:
    print(f"[app] Warning: {e}")
    _model = None
    _feature_importance = {}

_cache: dict[str, tuple[float, dict]] = {}


@app.exception_handler(Exception)
def handle_unexpected_error(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": f"Unexpected server error: {exc}"})


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/score/{ticker}")
def score(ticker: str):
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded — run `python src/train_model.py` first.",
        )

    ticker = ticker.upper()
    now = time.time()

    cached = _cache.get(ticker)
    if cached is not None and now - cached[0] < CACHE_TTL_SECONDS:
        return cached[1]

    result = predict(ticker, model=_model)

    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    result = {
        **result,
        "feature_importance": _feature_importance,
        "explanation": explain(result["scores"]),
    }
    _cache[ticker] = (now, result)
    return result
