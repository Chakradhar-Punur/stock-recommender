import yfinance as yf
import pandas as pd

def get_stock_data(ticker, start_date, end_date):
    """
    Fetch historical stock data for a given ticker symbol between specified dates.

    Parameters:
    ticker (str): The stock ticker symbol (e.g., 'AAPL' for Apple).
    start_date (str): The start date in 'YYYY-MM-DD' format.
    end_date (str): The end date in 'YYYY-MM-DD' format.

    Returns:
    pd.DataFrame: A DataFrame containing the historical stock data.
    """
    try:
        # Fetch the stock data using yfinance
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        
        # Check if data is returned
        if stock_data.empty:
            print(f"No data found for {ticker} between {start_date} and {end_date}.")
            return pd.DataFrame()  # Return an empty DataFrame
        
        return stock_data
    except Exception as e:
        print(f"An error occurred while fetching data for {ticker}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error


if __name__ == "__main__":
    # Example usage
    ticker = "AAPL"
    start_date = "2020-01-01"
    end_date = "2020-12-31"
    
    data = get_stock_data(ticker, start_date, end_date)
    print(data.head())  # Print the first few rows of the data