import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import os
from matplotlib import pyplot as plt
import math


class DataFetcher:
    """
    Fetch historical data from yfinance and save it locally.
    """

    def __init__(self, tickers: list, start_date: datetime, end_date: datetime):
        """
        Initialize DataFetcher.

        Args:
        - tickers (list): List of stock/index tickers.
        - start_date (datetime): Start date.
        - end_date (datetime): End date.
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.base_path = os.getcwd()
        self.data_dir = os.path.join(self.base_path, "Data")
        os.makedirs(self.data_dir, exist_ok=True)

    def fetch_and_save_data(self):
        """
        Fetch historical data for each ticker and save as CSV.
        """
        file_path = os.path.join(self.data_dir, "Index_data.csv")
        if os.path.exists(file_path):
            close = pd.read_csv(file_path)

        else:
            data = yf.download(self.tickers, start=self.start_date, end=self.end_date)['Adj Close']
            data = data.loc[:, self.tickers]
            data.ffill(inplace=True)
            data.to_csv(file_path)

        if os.path.exists(file_path):
             close = pd.read_csv(file_path, index_col=0)

        self.data = close.pct_change().dropna(how='any')

    def plot_distribuion(self):
        for ticker in self.tickers:
            return_data = np.array(self.data[ticker])
            plt.figure(figsize=(8, 6), dpi=100)
            plt.hist(return_data, bins=50, density=True, color='skyblue', edgecolor='black')
            plt.title(ticker)
            plt.plot()




# def generate_normal_bm(miu, sigma, n):
#     # generate D ~ Exp(1 / 2)
#     d = - 2 * np.log(np.random.uniform(0, 1, int(n / 2)))
#
#     # generate Θ ~ Unif(0, 2Π)
#     theta = 2 * math.pi * np.random.uniform(0, 1, int(n / 2))
#
#     # generate X, Y ~ Normal(miu, sigma)
#     X = np.sqrt(d) * np.cos(theta) * sigma + miu
#     Y = np.sqrt(d) * np.sin(theta) * sigma + miu
#     normal_random_variables = np.hstack((X, Y))
#     return normal_random_variables


# parameter_dict = {}
# distribution_dict = {}
# n = 100



    # assume all the returns satisfies gaussian distribution, estimate the parameters
    # miu = np.mean(return_data)
    # sigma = np.std(return_data, ddof=0)
    # parameter_dict[ticker] = [miu, sigma]
    # # generate random variables from the estimated distribution
    # random_number = generate_normal_bm(miu, sigma, n)
    # distribution_dict[ticker] = random_number.copy()





