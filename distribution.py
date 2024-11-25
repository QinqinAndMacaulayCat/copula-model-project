"""this module is used to save the functions and classes related to the distribution of the data, including:
    1. Multivariate: 
        - Empirical CDF: calculate the empirical CDF of the data
        - PPF
        - Graph
"""

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

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

 



