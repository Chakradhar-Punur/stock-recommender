"""
build_training_data.py
-----------------------
Builds the labeled dataset the ML model (train_model.py) will learn from.

For each ticker, we walk back through its price history and take a
snapshot every ~3 months. At each snapshot date T we compute the same
features a live prediction would use (momentum, valuation), then look
12 months into T's future to see whether the stock's return beat a 10%
threshold. That (features_at_T, beat_threshold) pair is one training row.

IMPORTANT — this file deliberately does NOT split into train/test. We
sample points chronologically across many years, but the split into
"train" vs "test" happens later in train_model.py, sorted by date. Doing
it there (not here) keeps this file focused on one job — assembling raw
labeled data — and keeps the leakage-sensitive split logic in one place.

KNOWN SIMPLIFICATION (worth calling out in an interview): yfinance's free
`.info` endpoint only exposes *current* fundamentals, not point-in-time
historical P/E. A fully rigorous dataset would use the P/E that was
actually true at each historical snapshot date. Here we approximate by
using each ticker's *current* P/E as a static valuation feature repeated
across all of that ticker's historical rows. This is a real limitation,
not an oversight — a production version would pull point-in-time
fundamentals from a paid data vendor (e.g. Compustat, FactSet, or a
fundamentals API with history).
"""

import time
import pandas as pd

from data_collection import get_stock_data
from features import calculate_momentum_score, calculate_valuation_score


# ~10 large-cap tickers spanning a few sectors (tech, financials, health,
# consumer staples, energy) so the model doesn't just learn "tech go up".
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "JPM", "JNJ", "PG", "XOM",
]

SAMPLE_FREQUENCY_MONTHS = 3   # snapshot every ~3 months
LOOKBACK_YEARS = 8            # how far back to sample from
FORWARD_MONTHS = 12           # label horizon: return over the following year
RETURN_THRESHOLD = 0.10       # "beat the market" bar: +10% over 12 months
FETCH_PERIOD = "10y"          # need LOOKBACK_YEARS + FORWARD_MONTHS of history


def _price_row_asof(hist: pd.DataFrame, date: pd.Timestamp) -> pd.Series | None:
    """
    Return the price row for the most recent trading day at-or-before
    `date` (markets are closed on weekends/holidays, so we can't just
    index by an arbitrary calendar date). Returns None if no such row
    exists (e.g. `date` is before the start of `hist`).
    """
    row = hist.asof(date)
    if row is None or pd.isna(row.get("Close")):
        return None
    return row


def build_training_dataset(
    tickers: list[str] = None,
    lookback_years: int = LOOKBACK_YEARS,
    sample_frequency_months: int = SAMPLE_FREQUENCY_MONTHS,
    forward_months: int = FORWARD_MONTHS,
    return_threshold: float = RETURN_THRESHOLD,
) -> pd.DataFrame:
    """
    Loop over `tickers`, sample price history every `sample_frequency_months`
    over the past `lookback_years`, and build labeled rows of
    (features at time T) -> (did the stock beat `return_threshold` return
    over the following `forward_months`?).

    Returns
    -------
    pd.DataFrame with columns:
        ticker, sample_date, momentum_score, valuation_score,
        forward_return, label   (label: 1 = beat threshold, 0 = did not)
    """
    if tickers is None:
        tickers = DEFAULT_TICKERS

    all_rows = []

    for ticker in tickers:
        print(f"[build_training_data] Fetching {ticker}...")
        data = get_stock_data(ticker, period=FETCH_PERIOD)
        hist = data["history"]
        info = data["info"]

        if hist.empty:
            print(f"[build_training_data] Skipping {ticker} — no price history.")
            continue

        # Static valuation proxy for this ticker (see module docstring —
        # this is today's P/E applied to every historical row, a known
        # simplification of the free data source).
        valuation_score = calculate_valuation_score(info)

        data_start = hist.index.min()
        data_end = hist.index.max()

        # Earliest a sample can be taken: needs >=30 days of prior history
        # for the momentum lookback. Latest: needs `forward_months` of
        # *future* data still available to compute the label, otherwise
        # we'd be guessing rather than labeling.
        sample_start = max(
            data_start + pd.Timedelta(days=35),
            data_end - pd.DateOffset(years=lookback_years),
        )
        sample_end = data_end - pd.DateOffset(months=forward_months)

        if sample_start >= sample_end:
            print(f"[build_training_data] Skipping {ticker} — not enough "
                  f"history to sample (need {lookback_years}y lookback + "
                  f"{forward_months}mo forward window).")
            continue

        ticker_row_count = 0
        sample_date = sample_start
        while sample_date <= sample_end:
            row_at_T = _price_row_asof(hist, sample_date)
            if row_at_T is not None:
                actual_T = row_at_T.name  # actual trading-day timestamp used
                hist_upto_T = hist.loc[:actual_T]
                price_T = row_at_T["Close"]

                future_date = actual_T + pd.DateOffset(months=forward_months)
                row_future = _price_row_asof(hist, future_date)

                # Guard against `asof` silently returning a stale (too-early)
                # row when we don't actually have `forward_months` of future
                # data yet — only accept rows genuinely close to the target date.
                has_real_future_data = (
                    row_future is not None
                    and hist.index.max() >= future_date - pd.Timedelta(days=10)
                )

                if has_real_future_data and price_T > 0:
                    price_future = row_future["Close"]
                    forward_return = (price_future - price_T) / price_T
                    label = int(forward_return >= return_threshold)
                    momentum_score = calculate_momentum_score(hist_upto_T)

                    all_rows.append({
                        "ticker": ticker,
                        "sample_date": actual_T.date().isoformat(),
                        "momentum_score": momentum_score,
                        "valuation_score": valuation_score,
                        "forward_return": round(float(forward_return), 4),
                        "label": label,
                    })
                    ticker_row_count += 1

            sample_date += pd.DateOffset(months=sample_frequency_months)

        print(f"[build_training_data]   -> {ticker_row_count} labeled samples")

        # Be a polite API citizen — small pause between tickers to reduce
        # the chance of hitting Yahoo's rate limiting.
        time.sleep(0.5)

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = df.sort_values("sample_date").reset_index(drop=True)
    return df


if __name__ == "__main__":
    import os

    dataset = build_training_dataset()

    print(f"\n=== Training dataset summary ===")
    print(f"Total rows: {len(dataset)}")
    if not dataset.empty:
        print(f"Date range: {dataset['sample_date'].min()} to {dataset['sample_date'].max()}")
        print(f"Label balance (1 = beat +10% over 12mo):")
        print(dataset["label"].value_counts(normalize=True).round(3))
        print(f"\nRows per ticker:")
        print(dataset["ticker"].value_counts())

    out_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "processed", "training_data.csv"
    )
    out_path = os.path.abspath(out_path)
    dataset.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
