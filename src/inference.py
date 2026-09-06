import os
import pickle

from data_collection import get_stock_data
from features import calculate_momentum_score, calculate_valuation_score


MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models", "xgboost_v1.pkl")
)

FEATURE_COLUMNS = ["momentum_score", "valuation_score"]

LABEL_NAMES = {0: "SELL", 1: "BUY"}


def load_model(model_path: str = MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at {model_path}. "
            f"Run `python src/train_model.py` first to train and save one."
        )
    with open(model_path, "rb") as f:
        return pickle.load(f)


def predict(ticker: str, model=None) -> dict:
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
        return {"ticker": ticker, "recommendation": None, "confidence": None, "error": str(e)}
    except Exception as e:
        return {"ticker": ticker, "recommendation": None, "confidence": None,
                 "error": f"Inference failed for '{ticker}': {e}"}


if __name__ == "__main__":
    import sys

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
