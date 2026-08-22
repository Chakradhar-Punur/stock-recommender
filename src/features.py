"""
features.py
------------
Turns raw data from data_collection.py into normalized 0-1 "scores" that
a model (rule-based today, ML later) can combine. Normalizing to a common
[0, 1] range up front matters for two reasons:
  1. It makes scores directly comparable/combinable (e.g. averaging a
     momentum score with a valuation score is meaningless if they're on
     different scales).
  2. It's what most ML models (including XGBoost, later) expect as input
     to behave well and to keep feature importances interpretable.

Each function below is intentionally small and pure (data in, number out)
so it's trivial to unit test and to reuse identically in build_training_data.py.
"""

import pandas as pd


def calculate_momentum_score(hist: pd.DataFrame) -> float:
    """
    Score a stock's recent momentum based on its 30-trading-day price change.

    Why 30 days: short enough to capture a real recent trend, long enough
    to filter out single-day noise. This is a classic simple momentum
    factor used as a baseline in quant finance.

    Parameters
    ----------
    hist : pd.DataFrame
        Price history as returned by data_collection.get_stock_data()["history"].
        Must contain a "Close" column with at least 31 rows.

    Returns
    -------
    float in [0, 1] — 0.5 means flat (0% change), 1.0 means +20% or more
    over 30 days, 0.0 means -20% or worse. We clip at +/-20% because daily
    stock moves rarely exceed that over a month, and clipping keeps
    occasional outliers (e.g. earnings-day spikes) from swamping the score.

    Returns 0.5 (neutral) if there isn't enough history to compute a
    30-day change, rather than raising — callers shouldn't have to
    special-case short histories.
    """
    if hist is None or hist.empty or "Close" not in hist.columns:
        return 0.5

    if len(hist) < 31:
        # Not enough history for a 30-day lookback; neutral score is a
        # safer default than guessing off a shorter window.
        return 0.5

    price_30d_ago = hist["Close"].iloc[-31]
    price_now = hist["Close"].iloc[-1]

    if price_30d_ago == 0:
        return 0.5

    pct_change = (price_now - price_30d_ago) / price_30d_ago

    # Map [-20%, +20%] linearly onto [0, 1], clipping anything outside
    # that range to the nearest boundary.
    clipped = max(min(pct_change, 0.20), -0.20)
    score = (clipped + 0.20) / 0.40
    return round(float(score), 4)


def calculate_valuation_score(info: dict) -> float:
    """
    Score a stock's valuation based on trailing P/E ratio — lower P/E
    scores higher, reflecting the classic "cheaper relative to earnings
    is more attractive" value-investing heuristic. This is deliberately
    a naive single-factor view (no sector adjustment, no growth
    adjustment) — good enough for a baseline, and something we can
    critique/improve once real ML features get added.

    Parameters
    ----------
    info : dict
        Fundamentals dict as returned by data_collection.get_stock_data()["info"].
        Expected to have a "trailingPE" key (may be None/missing).

    Returns
    -------
    float in [0, 1]. We map P/E of 10 -> 1.0 (cheap) and P/E of 40 -> 0.0
    (expensive), clipping outside that band. 10-40 covers the typical
    range for large-cap stocks; a negative or missing P/E (e.g. a company
    with negative earnings) is treated as "unknown" and scored neutral
    (0.5) rather than penalized, since a negative P/E doesn't mean
    "infinitely expensive".
    """
    if not info:
        return 0.5

    pe = info.get("trailingPE")

    if pe is None or pe <= 0:
        return 0.5

    # Map [10, 40] linearly onto [1, 0] (inverted: lower P/E = higher score).
    clipped_pe = max(min(pe, 40), 10)
    score = 1 - ((clipped_pe - 10) / (40 - 10))
    return round(float(score), 4)


if __name__ == "__main__":
    # Standalone test: reuse data_collection to pull real MSFT data, then
    # compute and print both scores so we can eyeball they're sane
    # (momentum near 0.5 for flat, valuation reflecting current P/E).
    from data_collection import get_stock_data

    data = get_stock_data("MSFT", period="2y")

    momentum = calculate_momentum_score(data["history"])
    valuation = calculate_valuation_score(data["info"])

    print(f"=== {data['ticker']} feature scores ===")
    print(f"Momentum score (30-day price change): {momentum}")
    print(f"Valuation score (trailing P/E = {data['info'].get('trailingPE')}): {valuation}")
