import yfinance as yf
import pandas as pd


def get_stock_data(ticker: str, period: str = "2y") -> dict:
    try:
        stock = yf.Ticker(ticker)

        # `.history()` gives OHLCV bars; empty on a bad ticker rather than
        # raising, so we explicitly check `.empty` below instead of relying
        # on an exception.
        hist = stock.history(period=period)

        if hist.empty:
            print(f"[data_collection] No price history returned for '{ticker}'. "
                  f"Check the ticker symbol is valid.")
            return {"history": pd.DataFrame(), "info": {}, "ticker": ticker}

        # `.info` is a big dict scraped from Yahoo's site; it's the flakiest
        # part of yfinance (fields vary and can go missing entirely). We
        # pull it defensively and only keep the couple of fields we need
        # right now, using .get() so a missing key never raises KeyError.
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
        # Broad except is intentional here: yfinance can fail in many ways
        # (network error, rate limiting, malformed response) and this is a
        # leaf function we never want to let crash the whole pipeline.
        print(f"[data_collection] Failed to fetch data for '{ticker}': {e}")
        return {"history": pd.DataFrame(), "info": {}, "ticker": ticker}


if __name__ == "__main__":
    # Quick standalone sanity check: run `python src/data_collection.py`
    # and eyeball that real data comes back before wiring this into
    # anything else.
    result = get_stock_data("MSFT", period="2y")

    print(f"\n=== {result['ticker']} — last 5 rows of price history ===")
    print(result["history"].tail(5))

    print(f"\n=== {result['ticker']} — fundamentals ===")
    pe = result["info"].get("trailingPE")
    print(f"Trailing P/E ratio: {pe}")
    print(f"Market cap: {result['info'].get('marketCap')}")
