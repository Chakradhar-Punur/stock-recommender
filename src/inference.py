"""
inference.py
-------------
The "serving" layer of the pipeline: load the trained model once, then
score any ticker on demand. This is the piece a future API/UI layer would
call directly — everything here is deliberately decoupled from training
so inference stays fast and simple (no re-fitting, no CSV reads).
"""

import os
import pickle

from data_collection import get_stock_data
from features import calculate_momentum_score, calculate_valuation_score


MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models", "xgboost_v1.pkl")
)

# Must match FEATURE_COLUMNS in train_model.py exactly, in the same order —
# XGBoost doesn't know column names at predict time, only column position.
FEATURE_COLUMNS = ["momentum_score", "valuation_score"]

# Index of "BUY" (label == 1) in predict_proba's output columns. sklearn/
# XGBoost classifiers order columns by sorted class label, so for binary
# labels {0, 1} column 1 is always the positive class.
LABEL_NAMES = {0: "SELL", 1: "BUY"}


def load_model(model_path: str = MODEL_PATH):
    """
    Load the pickled XGBoost model from disk.

    Raises a clear, actionable error rather than a raw FileNotFoundError/
    pickle traceback if the model hasn't been trained yet — this is the
    kind of thing a teammate (or interviewer running your code) would hit
    immediately if they skip straight to inference.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at {model_path}. "
            f"Run `python src/train_model.py` first to train and save one."
        )
    with open(model_path, "rb") as f:
        return pickle.load(f)


def predict(ticker: str, model=None) -> dict:
    """
    Score a single ticker with the trained model.

    Parameters
    ----------
    ticker : str
    model : trained XGBoost classifier, optional
        Pass an already-loaded model to avoid re-reading the pickle file
        on every call (e.g. when scoring many tickers in a loop). If
        omitted, loads MODEL_PATH fresh.

    Returns
    -------
    dict with:
        "ticker"         : str
        "recommendation" : "BUY" or "SELL" (the model's binary predicted class)
        "confidence"     : float — predicted probability of that class, as a %
        "error"          : str, only present if something went wrong

    On any failure (bad ticker, no price data, model missing) this
    returns an "error" key instead of raising, so a caller scoring a
    batch of tickers can skip failures without the whole run crashing.
    """
    try:
        if model is None:
            model = load_model()

        data = get_stock_data(ticker)
        if data["history"].empty:
            return {
                "ticker": ticker,
                "recommendation": None,
                "confidence": None,
                "error": f"No price data available for '{ticker}'.",
            }

        momentum_score = calculate_momentum_score(data["history"])
        valuation_score = calculate_valuation_score(data["info"])

        # Model expects a 2D array: one row, columns in FEATURE_COLUMNS order.
        X = [[momentum_score, valuation_score]]

        predicted_class = int(model.predict(X)[0])
        class_probabilities = model.predict_proba(X)[0]
        confidence = float(class_probabilities[predicted_class]) * 100

        return {
            "ticker": ticker,
            "recommendation": LABEL_NAMES[predicted_class],
            "confidence": round(confidence, 1),
        }

    except FileNotFoundError as e:
        # Re-raised distinctly from other errors so a CLI caller can tell
        # "you forgot to train the model" apart from "this ticker is bad".
        return {"ticker": ticker, "recommendation": None, "confidence": None, "error": str(e)}
    except Exception as e:
        return {"ticker": ticker, "recommendation": None, "confidence": None,
                 "error": f"Inference failed for '{ticker}': {e}"}


if __name__ == "__main__":
    import sys

    # Allow `python src/inference.py TICKER` from the command line, default
    # to a small demo list if no ticker is given.
    tickers = sys.argv[1:] if len(sys.argv) > 1 else ["AAPL", "MSFT", "TSLA"]

    try:
        model = load_model()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    for t in tickers:
        result = predict(t, model=model)
        if "error" in result:
            print(f"{t}: ERROR — {result['error']}")
        else:
            print(f"{result['ticker']}: {result['recommendation']} "
                  f"(confidence: {result['confidence']}%)")
