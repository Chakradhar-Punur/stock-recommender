import pandas as pd

def calculate_momentum_score(hist: pd.DataFrame) -> float:
    if hist is None or hist.empty or "Close" not in hist.columns:
        return 0.5

    if len(hist) < 31:
        return 0.5

    price_30d_ago = hist["Close"].iloc[-31]
    price_now = hist["Close"].iloc[-1]

    if price_30d_ago == 0:
        return 0.5

    pct_change = (price_now - price_30d_ago) / price_30d_ago

    clipped = max(min(pct_change, 0.20), -0.20)
    score = (clipped + 0.20) / 0.40
    return round(float(score), 4)


def calculate_valuation_score(info: dict) -> float:
    if not info:
        return 0.5

    pe = info.get("trailingPE")

    if pe is None or pe <= 0:
        return 0.5

    clipped_pe = max(min(pe, 40), 10)
    score = 1 - ((clipped_pe - 10) / (40 - 10))
    return round(float(score), 4)


if __name__ == "__main__":
    from data_collection import get_stock_data

    data = get_stock_data("MSFT", period="2y")

    momentum = calculate_momentum_score(data["history"])
    valuation = calculate_valuation_score(data["info"])

    print(f"=== {data['ticker']} feature scores ===")
    print(f"Momentum score (30-day price change): {momentum}")
    print(f"Valuation score (trailing P/E = {data['info'].get('trailingPE')}): {valuation}")
