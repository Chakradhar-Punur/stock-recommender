"""
baseline_model.py
------------------
A deliberately simple, rule-based recommendation function. This is NOT the
ML model — it's the baseline we'll compare the trained XGBoost model
against later. Without a baseline, "our model gets 62% accuracy" is a
meaningless number; "our model beats a simple rule-based baseline by 8
points" is a claim you can actually defend in an interview.

The rule combines the two 0-1 scores from features.py with equal weights
into a single composite score, then thresholds it into BUY/HOLD/SELL.
Equal weighting is a starting assumption, not a tuned result — flagging
that explicitly here so it's obvious this is meant to be superseded.
"""

from features import calculate_momentum_score, calculate_valuation_score


# Thresholds are simple round numbers, not backtested — again, intentional
# for a baseline: it should be as "dumb" and transparent as possible so
# any improvement from the ML model is a fair, honest comparison.
BUY_THRESHOLD = 0.65
SELL_THRESHOLD = 0.35


def recommend(momentum_score: float, valuation_score: float) -> dict:
    """
    Combine momentum + valuation scores into a BUY/HOLD/SELL call.

    Parameters
    ----------
    momentum_score : float in [0, 1]
    valuation_score : float in [0, 1]

    Returns
    -------
    dict with:
        "composite_score" : float — simple average of the two inputs
        "recommendation"  : str   — "BUY", "HOLD", or "SELL"

    Why a simple average and fixed thresholds: this is the baseline a
    future ML model needs to beat. If the baseline were already clever
    (weighted, tuned thresholds, extra factors), it would set an unfairly
    high bar and muddy the comparison. Keep it dumb on purpose.
    """
    composite_score = round((momentum_score + valuation_score) / 2, 4)

    if composite_score >= BUY_THRESHOLD:
        recommendation = "BUY"
    elif composite_score <= SELL_THRESHOLD:
        recommendation = "SELL"
    else:
        recommendation = "HOLD"

    return {
        "composite_score": composite_score,
        "recommendation": recommendation,
    }


if __name__ == "__main__":
    # Standalone test: pull real MSFT data through the full chain
    # (data_collection -> features -> baseline_model) and print the result.
    from data_collection import get_stock_data

    data = get_stock_data("MSFT", period="2y")
    momentum = calculate_momentum_score(data["history"])
    valuation = calculate_valuation_score(data["info"])

    result = recommend(momentum, valuation)

    print(f"=== {data['ticker']} baseline recommendation ===")
    print(f"Momentum score:   {momentum}")
    print(f"Valuation score:  {valuation}")
    print(f"Composite score:  {result['composite_score']}")
    print(f"Recommendation:   {result['recommendation']}")
