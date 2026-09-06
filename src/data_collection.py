import yfinance as yf
import pandas as pd


def get_stock_data(ticker: str, period: str = "2y") -> dict:
    try:
        stock = yf.Ticker(ticker)

        hist = stock.history(period=period)

        if hist.empty:
            print(f"[data_collection] No price history returned for '{ticker}'. "
                  f"Check the ticker symbol is valid.")
            return {"history": pd.DataFrame(), "info": {}, "ticker": ticker}

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
        }

        return {"history": hist, "info": info, "ticker": ticker}

    except Exception as e:
        print(f"[data_collection] Failed to fetch data for '{ticker}': {e}")
        return {"history": pd.DataFrame(), "info": {}, "ticker": ticker}


if __name__ == "__main__":
    result = get_stock_data("MSFT", period="2y")

    print(f"\n=== {result['ticker']} — last 5 rows of price history ===")
    print(result["history"].tail(5))

    print(f"\n=== {result['ticker']} — fundamentals ===")
    pe = result["info"].get("trailingPE")
    print(f"Trailing P/E ratio: {pe}")
    print(f"Market cap: {result['info'].get('marketCap')}")
