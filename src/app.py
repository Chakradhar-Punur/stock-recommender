"""
app.py
------
Thin FastAPI wrapper around inference.py. This file has zero ML/business
logic of its own on purpose — it only translates HTTP requests into calls
to the already-tested `predict()` function and serializes the result back
to JSON. Keeping it thin means the API layer can't silently diverge from
what inference.py actually does, and it's trivial to swap this for
another framework later without touching the model-serving code.

Run with (from the project root):
    cd src && ../venv/bin/uvicorn app:app --reload --port 8000
(run from inside src/, matching how inference.py imports its sibling
modules — data_collection.py, features.py — with plain flat imports
rather than package-relative ones)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from inference import predict, load_model

app = FastAPI(title="Stock Recommender API")

# CORS: the frontend is a plain HTML file opened via file:// or served by
# a separate static server, not this same origin — without this, the
# browser blocks the fetch() call with a CORS error before it ever
# reaches this server. Wide open ("*") is fine for a local demo; a real
# deployment would restrict this to the frontend's actual origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model once at startup rather than on every request — reading
# and unpickling the file per-request would add latency for no benefit,
# since the model doesn't change between requests.
try:
    _model = load_model()
except FileNotFoundError as e:
    # Don't crash the whole server if the model hasn't been trained yet;
    # let it start so /  still works, and fail informatively per-request
    # in /score instead (see below).
    print(f"[app] Warning: {e}")
    _model = None


@app.get("/")
def root():
    """Liveness check — hit this first to confirm the server is up."""
    return {"status": "ok"}


@app.get("/score/{ticker}")
def score(ticker: str):
    """
    Score a ticker and return its recommendation.

    Delegates entirely to inference.predict(), which already returns an
    "error" key instead of raising on bad tickers/missing data — we
    translate that into a proper HTTP error status here so the frontend
    gets a real failed fetch() rather than a 200 with an error buried
    inside it.
    """
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded — run `python src/train_model.py` first.",
        )

    result = predict(ticker.upper(), model=_model)

    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    return result
