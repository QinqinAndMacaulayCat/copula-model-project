import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import os


class DataFetcher:
    """
    Fetch historical data from yfinance and save it locally.
    """

    def __init__(self, tickers: list, start_date: datetime, end_date: datetime, interval: str = "1d"):
        """
        Initialize DataFetcher.

        Args:
        - tickers (list): List of stock/index tickers.
        - start_date (datetime): Start date.
        - end_date (datetime): End date.
        - interval (str): Data interval (default "1d").
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.base_path = os.getcwd()
        self.data_dir = os.path.join(self.base_path, "Data")
        os.makedirs(self.data_dir, exist_ok=True)

    def fetch_and_save_data(self):
        """
        Fetch historical data for each ticker and save as CSV.
        """
        print(f"Data for tickers {', '.join(self.tickers)} saved to {self.data_dir}.")
        print()

        
        for ticker in self.tickers:
            print(f"Fetching data for {ticker}...")
            stock = yf.Ticker(ticker)
            df = stock.history(start=self.start_date, end=self.end_date, interval=self.interval)

            if df.empty:
                print(f"No data found for {ticker}. Skipping.")
                continue

            df = df.drop(columns=['Dividends', 'Stock Splits'], errors='ignore')
            safe_ticker = ticker.replace('^', '')
            file_path = os.path.join(self.data_dir, f"{safe_ticker}_data.csv")
            df.to_csv(file_path)
            
        print()

            

    def load_and_calculate_returns(self) -> pd.DataFrame:
        """
        Load saved CSVs and calculate returns.

        Returns:
        - pd.DataFrame: Combined returns data.
        """
        files = [os.path.join(self.data_dir, f"{ticker.replace('^', '')}_data.csv") for ticker in self.tickers]
        dataframes = {}

        for file in files:
            asset_name = os.path.basename(file).replace('_data.csv', '')
            df = pd.read_csv(file, parse_dates=['Date'], index_col='Date')
            df['Returns'] = 100 * df['Close'].pct_change()
            df['log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
            df.dropna(inplace=True)
            dataframes[asset_name] = df
            print(f"Processed data for {asset_name}, shape: {df.shape}.")

        returns_data = {name: df['Returns'] for name, df in dataframes.items()}
        combined_data = pd.concat(returns_data, axis=1).dropna()
        combined_data.columns = dataframes.keys()

        return combined_data
