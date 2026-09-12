import os
import time
from typing import cast

import pandas as pd

from data_collection import get_stock_data
from scoring.valuation import calculate_valuation_score
from scoring.growth import calculate_growth_score
from scoring.momentum import calculate_momentum_score
from scoring.quality import calculate_quality_score
from scoring.risk import calculate_risk_score


DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "JPM", "JNJ", "PG", "XOM",
]

SAMPLE_FREQUENCY_MONTHS = 3
LOOKBACK_YEARS = 8
FORWARD_MONTHS = 12
RETURN_THRESHOLD = 0.10
FETCH_PERIOD = "10y"


def _price_row_asof(hist: pd.DataFrame, date: pd.Timestamp) -> pd.Series | None:
    row = hist.asof(date)
    if not isinstance(row, pd.Series):
        return None
    if bool(pd.isna(row.get("Close"))):
        return None
    return row


def build_training_dataset(
    tickers: list[str] | None = None,
    lookback_years: int = LOOKBACK_YEARS,
    sample_frequency_months: int = SAMPLE_FREQUENCY_MONTHS,
    forward_months: int = FORWARD_MONTHS,
    return_threshold: float = RETURN_THRESHOLD,
) -> pd.DataFrame:
    if tickers is None:
        tickers = DEFAULT_TICKERS

    all_rows = []

    for ticker in tickers:
        print(f"[build_training_data] Fetching {ticker}...")
        data = get_stock_data(ticker, period=FETCH_PERIOD)
        hist = data["history"]
        info = data["info"]
        financials = data["financials"]

        if hist.empty:
            print(f"[build_training_data] Skipping {ticker} — no price history.")
            continue

        valuation_score = calculate_valuation_score(info)
        growth_score = calculate_growth_score(financials)
        quality_score = calculate_quality_score(info)

        data_start = hist.index.min()
        data_end = hist.index.max()

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
                actual_T = cast(pd.Timestamp, row_at_T.name)
                hist_upto_T = hist.loc[:actual_T]
                price_T = row_at_T["Close"]

                future_date = actual_T + pd.DateOffset(months=forward_months)
                row_future = _price_row_asof(hist, future_date)

                if (
                    row_future is not None
                    and hist.index.max() >= future_date - pd.Timedelta(days=10)
                    and price_T > 0
                ):
                    price_future = row_future["Close"]
                    forward_return = (price_future - price_T) / price_T
                    label = int(forward_return >= return_threshold)
                    momentum_score = calculate_momentum_score(hist_upto_T)
                    risk_score = calculate_risk_score(hist_upto_T, info)

                    all_rows.append({
                        "ticker": ticker,
                        "sample_date": actual_T.date().isoformat(),
                        "valuation_score": valuation_score,
                        "growth_score": growth_score,
                        "momentum_score": momentum_score,
                        "quality_score": quality_score,
                        "risk_score": risk_score,
                        "forward_return": round(float(forward_return), 4),
                        "label": label,
                    })
                    ticker_row_count += 1

            sample_date += pd.DateOffset(months=sample_frequency_months)

        print(f"[build_training_data]   -> {ticker_row_count} labeled samples")
        time.sleep(0.5)

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = df.sort_values("sample_date").reset_index(drop=True)
    return df


if __name__ == "__main__":
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
