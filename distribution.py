"""this module is used to save the functions and classes related to the distribution of the data, including:
    1. Multivariate: 
        - Empirical CDF: calculate the empirical CDF of the data
        - PPF
        - Graph
"""

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Multivariate:
    def __init__(self, data):
        self.data = data
        self.cov = data.cov()
        self.corr = data.corr()
    
    def empircal_cdf(self):
        self.ecdf = self.data.rank() / len(self.data)
        print("rank", self.data.rank(), "len", len(self.data))
        return self.ecdf

    def ppf(self, u):
        """
        u: np.array, the quantile
        """
        pass
    
    def empircal_ppf(self, u):
        """
        u: np.array, the quantile
        """
        ppfs = []
        for i in range(self.data.shape[1]):
            ppf_value = np.quantile(self.data.iloc[:, i], u[:, i])
            ppfs.append(ppf_value)

        ppfs = np.array(ppfs).T

        return ppfs
    
    @staticmethod
    def extreme_value_correlation(df, percentile=95, direction="upper"):
        """
        Calculate the extreme value correlation (based on extracting tail data using percentiles). We simply compute the percentage of two columns that are both extreme given one is extreme. 
        :param df: DataFrame, The dataset
        :param percentile: The percentile threshold (e.g., 95)
        :param direction: The direction of the extreme values ("upper" or "lower")
        :return: The correlation of extreme values. i row j column means the j column is extreme given the i column is extreme.
        """
        # Compute thresholds for each column
        thresholds = {col: np.percentile(df[col].dropna(), percentile) for col in df.columns}
        # Initialize correlation matrix
        columns = df.columns
        corr_matrix = pd.DataFrame(index=columns, columns=columns, dtype=float)
    
        # Compute pairwise extreme correlations
        for col1 in columns:
            for col2 in columns:
                # Extract the threshold values
                threshold1 = thresholds[col1]
                threshold2 = thresholds[col2]
            
                # Filter rows where both columns exceed their respective thresholds
                if direction == "upper":
                    extreme_mask1 = df[col1] >= threshold1   
                    extreme_mask = (df[col1] >= threshold1) & (df[col2] >= threshold2)
                elif direction == "lower":
                    extreme_mask1 = df[col1] <= threshold1
                    extreme_mask = (df[col1] <= threshold1) & (df[col2] <= threshold2)
                else:   
                    raise ValueError("Invalid direction. Choose 'upper' or 'lower'.")
                corr_matrix.loc[col1, col2] = sum(extreme_mask) / sum(extreme_mask1)
        
        return corr_matrix
        
    @staticmethod
    def heatmap(data, title):
        """
        Plot a heatmap of the given data.
        :param data: DataFrame, The data to plot.
        :param title: str, The title of the plot.
        """
        plt.figure(figsize=(10, 10))
        sns.heatmap(data, 
                    annot=True,
                    fmt=".2f",
                    cmap="YlGnBu",
                    linewidths=0.5,
                    linecolor="gray",)
        plt.title(title, fontsize=15)
        plt.savefig(f"result/{title}.png")

    @staticmethod
    def plot_kde_comparison(df, title):

        sns.pairplot(df, kind="scatter", diag_kind="kde", plot_kws={'alpha': 0.7})
        plt.suptitle(title, y=1.02)  

        plt.savefig(f"result/{title}.png")
