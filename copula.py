import numpy as np

import pandas as pd
from scipy.optimize import minimize
from distribution import Multivariate
from scipy.stats import norm


def mypower(x, y):
    """
    use different method to calculate the power of x and y to avoid overflow.
    return: np.array, the power of x and y.
    """
    x = np.clip(x, 1e-10, 1e10)
    log_x = np.log(x)
    power = np.exp(y * log_x)

    return power


class Clayton:
    def __init__(self):
        self.theta = 0
        self.bound = (1e-5, 1)

    def c(self, u: np.ndarray, v: np.ndarray):
        """
        return: np.array, the density of Clayton copula
        """
        return (1 + self.theta) * mypower(u * v, -1 - self.theta) \
            * mypower(mypower(u, -self.theta) + mypower(v, -self.theta) - 1, -2 - 1 / self.theta)

    def h(self, u: np.ndarray, v: np.ndarray):
        """

        return: np.array, the h function/partial derivative F(u|v)  of Clayton copula
        since h function is basically a kind of conditional CDF, it should be between 0 and 1.

        """
        a = mypower(v, -self.theta - 1)
        b = mypower(u, -self.theta) + mypower(v, -self.theta) - 1
        c = mypower(b, -1 - 1 / self.theta)
        result = a * c

        # todo check which theta value will lead to nan value.
        if self.theta > 1000:
            result[np.isnan(result)] = 1
        else:
            result[np.isnan(result)] = 0

        return result

    def inverse_h(self, w: np.ndarray, v: np.ndarray):

        """
        return: np.array, the inverse of h function, which is the conditional CDF of u given v.
        since the inverse of h function will lead to the x, which is uniform distributed, the value should be between 0 and 1.
        """

        a = w * mypower(v, self.theta + 1)
        b = mypower(a, -self.theta / (1 + self.theta))
        c = mypower(v, -self.theta)
        d = mypower(b + 1 - c, -1 / self.theta)

        # todo: to avoid the nan value we add this adjustment here. when the correlation is extremely high, the value of the u (inverse of h function) should be equal to v, else the values should almost be independent of v.
        if self.theta > 1000:
            d[np.isnan(d)] = v[np.isnan(d)]
        else:
            d[np.isnan(d)] = w[np.isnan(d)]

        return d


class Gaussian:
    def __init__(self):
        self.theta = 0
        self.bound = (1e-5, 1)

    def c(self, u, v):
        """
        return the density of Clayton copula
        """
        return (1 + np.sqrt(1 - mypower(self.theta, 2))) * np.exp(-(mypower(self.theta, 2) * (u ** 2 + v ** 2) - 2 * self.theta * u * v) / \
                                                                  2 * (1 - mypower(self.theta, 2)))

    def h(self, u, v):
        """
        return the h function/partial derivative F(u|v)  of Clayton copula

        """

        a = (norm.ppf(u) - self.theta * norm.ppf(v)) / np.sqrt(1 - self.theta ** 2)

        return norm.cdf(a)

    def inverse_h(self, w, v):
        """
        return the inverse of h function, which is the conditional CDF of u given v.
        """

        a = norm.ppf(w) * np.sqrt(1 - self.theta ** 2) + self.theta * norm.ppf(v)

        return norm.cdf(a)
