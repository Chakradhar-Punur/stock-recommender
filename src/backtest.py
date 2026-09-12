import os
from typing import cast

import numpy as np
import pandas as pd
import yfinance as yf

from train_model import build_model, load_dataset, FEATURE_COLUMNS, TARGET_COLUMN

MIN_TRAIN_DATES = 8
SAMPLES_PER_YEAR = 4
SP500_TICKER = "^GSPC"


def walk_forward_predict(df: pd.DataFrame, min_train_dates: int = MIN_TRAIN_DATES) -> pd.DataFrame:
    unique_dates = sorted(df["sample_date"].unique())
    if len(unique_dates) <= min_train_dates:
        raise ValueError(
            f"Need more than {min_train_dates} unique sample dates to backtest; got {len(unique_dates)}."
        )

    rows = []
    for test_date in unique_dates[min_train_dates:]:
        train_df = df[df["sample_date"] < test_date]
        test_df = df[df["sample_date"] == test_date]

        model = build_model()
        model.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN])
        predicted = model.predict(test_df[FEATURE_COLUMNS])

        for (_, row), pred in zip(test_df.iterrows(), predicted):
            rows.append({
                "sample_date": row["sample_date"],
                "ticker": row["ticker"],
                "predicted_label": int(pred),
                "actual_label": int(cast(int, row[TARGET_COLUMN])),
                "forward_return": row["forward_return"],
            })

    return pd.DataFrame(rows)


def compute_strategy_returns(results: pd.DataFrame) -> pd.Series:
    buy_signals = results[results["predicted_label"] == 1]
    per_date_return = cast(
        pd.Series, buy_signals.groupby("sample_date")["forward_return"].mean()
    )
    return per_date_return.reindex(sorted(results["sample_date"].unique()), fill_value=0.0)


def compute_sharpe(returns: pd.Series, periods_per_year: int = SAMPLES_PER_YEAR) -> float:
    values = np.asarray(returns, dtype=float)
    std = float(values.std())
    if std == 0:
        return 0.0
    mean = float(values.mean())
    return float((mean / std) * np.sqrt(periods_per_year))


def compute_max_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns).cumprod()
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    return float(drawdown.min())


def non_overlapping_annual_returns(strategy_returns: pd.Series) -> pd.Series:
    df = strategy_returns.reset_index()
    df.columns = pd.Index(["sample_date", "return"])
    df["year"] = pd.to_datetime(df["sample_date"]).dt.year
    first_per_year = df.groupby("year").first()
    return cast(pd.Series, first_per_year["return"])


def fetch_sp500_history(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return yf.Ticker(SP500_TICKER).history(
        start=start.date().isoformat(), end=end.date().isoformat()
    )


def sp500_buy_hold_return(hist: pd.DataFrame) -> float:
    if hist.empty:
        return float("nan")
    return float(hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1)


def sp500_yearly_returns(hist: pd.DataFrame) -> dict:
    if hist.empty:
        return {}
    yearly = {}
    years = [ts.year for ts in cast(pd.DatetimeIndex, hist.index)]
    for year, group in hist.groupby(years):
        if len(group) < 2:
            continue
        yearly[int(cast(int, year))] = float(group["Close"].iloc[-1] / group["Close"].iloc[0] - 1)
    return yearly


def yearly_breakdown(results: pd.DataFrame, sp500_yearly: dict) -> pd.DataFrame:
    results = results.copy()
    results["year"] = pd.to_datetime(results["sample_date"]).dt.year

    rows = []
    for year_key, group in results.groupby("year"):
        year = int(cast(int, year_key))
        buy_signals = group[group["predicted_label"] == 1]
        accuracy = float(cast(float, (group["predicted_label"] == group["actual_label"]).mean()))

        if buy_signals.empty:
            win_rate = None
            avg_buy_return = None
        else:
            win_rate = round(float(cast(float, (buy_signals["forward_return"] > 0).mean())), 3)
            avg_buy_return = round(float(cast(float, buy_signals["forward_return"].mean())), 3)

        rows.append({
            "year": year,
            "samples": len(group),
            "buy_signals": len(buy_signals),
            "accuracy": round(accuracy, 3),
            "win_rate": win_rate,
            "avg_buy_forward_return": avg_buy_return,
            "sp500_return": round(sp500_yearly[year], 3) if year in sp500_yearly else None,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    csv_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "processed", "training_data.csv")
    )
    if not os.path.exists(csv_path):
        print(f"[backtest] No training data found at {csv_path}. Run build_training_data.py first.")
        raise SystemExit(1)

    dataset = load_dataset(csv_path)
    results = walk_forward_predict(dataset)

    overall_accuracy = float(cast(float, (results["predicted_label"] == results["actual_label"]).mean()))
    buy_signals = results[results["predicted_label"] == 1]
    overall_win_rate = (
        float(cast(float, (buy_signals["forward_return"] > 0).mean()))
        if not buy_signals.empty else None
    )

    strategy_returns = compute_strategy_returns(results)
    naive_sharpe = compute_sharpe(strategy_returns)
    naive_max_dd = compute_max_drawdown(strategy_returns)

    annual_returns = non_overlapping_annual_returns(strategy_returns)
    annual_sharpe = compute_sharpe(annual_returns, periods_per_year=1)
    annual_max_dd = compute_max_drawdown(annual_returns)
    annual_total_return = float((1 + annual_returns).prod() - 1)

    min_sample_date = cast(pd.Timestamp, pd.to_datetime(results["sample_date"]).min())
    max_sample_date = cast(pd.Timestamp, pd.to_datetime(results["sample_date"]).max())
    start_date = min_sample_date
    end_date = max_sample_date + pd.DateOffset(months=12)
    sp500_hist = fetch_sp500_history(start_date, end_date)
    sp500_return = sp500_buy_hold_return(sp500_hist)
    sp500_yearly = sp500_yearly_returns(sp500_hist)

    print("=== Walk-forward backtest ===")
    print(f"Test period: {start_date.date()} to {max_sample_date.date()}")
    print(f"Out-of-sample predictions: {len(results)}")
    print(f"Overall accuracy: {overall_accuracy:.3f}")
    print(f"BUY signals issued: {len(buy_signals)}")
    if overall_win_rate is not None:
        print(f"Win rate on BUY signals: {overall_win_rate:.3f}")
    else:
        print("Win rate on BUY signals: n/a (no BUY signals issued)")

    print(f"\n[NAIVE — overlapping 12mo windows sampled quarterly, inflated, for reference only]")
    print(f"  Sharpe ratio (x sqrt({SAMPLES_PER_YEAR})): {naive_sharpe:.3f}")
    print(f"  Max drawdown: {naive_max_dd:.3f}")

    print(f"\n[HONEST — one non-overlapping return per calendar year]")
    print(f"  Annual Sharpe ratio: {annual_sharpe:.3f}")
    print(f"  Max drawdown: {annual_max_dd:.3f}")
    print(f"  Compounded total return over period: {annual_total_return:.3f}")
    print(f"  S&P 500 buy-and-hold return, same period: {sp500_return:.3f}")

    print("\n=== Year-by-year breakdown ===")
    breakdown = yearly_breakdown(results, sp500_yearly)
    print(breakdown.to_string(index=False))

    out_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "processed", "backtest_results.csv")
    )
    results.to_csv(out_path, index=False)
    print(f"\nRaw out-of-sample predictions saved to {out_path}")
