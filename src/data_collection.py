import yfinance as yf
import pandas as pd


def get_stock_data(ticker: str, period: str = "2y") -> dict:
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)

        if hist.empty:
            print(f"[data_collection] No price history returned for '{ticker}'. "
                  f"Check the ticker symbol is valid.")
            return {"history": pd.DataFrame(), "info": {}, "financials": {}, "ticker": ticker}

        try:
            raw_info = stock.info
        except Exception as info_err:
            print(f"[data_collection] Warning: could not fetch fundamentals for "
                  f"'{ticker}' ({info_err}). Continuing with price history only.")
            raw_info = {}

        info = {
            "trailingPE": raw_info.get("trailingPE"),
            "forwardPE": raw_info.get("forwardPE"),
            "marketCap": raw_info.get("marketCap"),
            "longName": raw_info.get("longName", ticker),
            # valuation.py
            "pegRatio": raw_info.get("pegRatio"),
            "earningsGrowth": raw_info.get("earningsGrowth"),
            "debtToEquity": raw_info.get("debtToEquity"),
            # quality.py
            "returnOnEquity": raw_info.get("returnOnEquity"),
            "profitMargins": raw_info.get("profitMargins"),
            "freeCashflow": raw_info.get("freeCashflow"),
            "totalRevenue": raw_info.get("totalRevenue"),
            # risk.py
            "beta": raw_info.get("beta"),
        }

        financials = _get_quarterly_financials(stock)

        return {"history": hist, "info": info, "financials": financials, "ticker": ticker}
    
    except Exception as e:
        print(f"[data_collection] Failed to fetch data for '{ticker}': {e}")
        return {"history": pd.DataFrame(), "info": {}, "financials": {}, "ticker": ticker}

def _get_quarterly_financials(stock: yf.Ticker) -> dict:
    try:
        financials = stock.quarterly_financials
        if financials is None or financials.empty:
            return {"quarterly_revenue": [], "quarterly_earnings": []}
        return {
            "quarterly_revenue": _extract_row(financials, ["Total Revenue"]),
            "quarterly_earnings": _extract_row(financials, ["Net Income", "Net Income Common Stockholders"]),
        }
    except Exception as e:
        print(f"[data_collection] Warning: could not fetch quarterly financials for "
              f"'{stock.ticker}' ({e}).")
        return {"quarterly_revenue": [], "quarterly_earnings": []}

def _extract_row(financials: pd.DataFrame, row_names: list[str]) -> list[float]:
    """
    Try each candidate row label in turn (line-item names vary slightly
    across tickers/yfinance versions) and return its values oldest-to-
    newest (yfinance returns columns newest-first, so reverse). Only
    give up (return []) once every candidate has been tried — the
    previous version returned [] after the *first* miss, which defeated
    the point of passing a fallback list like ["Net Income",
    "Net Income Common Stockholders"].
    """
    for name in row_names:
        if name in financials.index:
            return list(reversed(financials.loc[name].tolist()))
    return []
