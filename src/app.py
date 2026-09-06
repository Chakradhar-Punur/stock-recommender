"""
Run with (from the project root):
    cd src && ../venv/bin/uvicorn app:app --reload --port 8000
"""
 
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from inference import predict, load_model

app = FastAPI(title="Stock Recommender API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    _model = load_model()
except FileNotFoundError as e:
    print(f"[app] Warning: {e}")
    _model = None


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/score/{ticker}")
def score(ticker: str):
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded — run `python src/train_model.py` first.",
        )

    result = predict(ticker.upper(), model=_model)

    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    return result
