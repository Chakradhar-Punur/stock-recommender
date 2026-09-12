from features import calculate_momentum_score, calculate_valuation_score


BUY_THRESHOLD = 0.65
SELL_THRESHOLD = 0.35


def recommend(momentum_score: float, valuation_score: float) -> dict:
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
