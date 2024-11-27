import numpy as np
from scipy.stats import norm


def mypower(x, y):
    """
    use different method to calculate
    the power of x and y to avoid overflow.
    return: np.array, the power of x and y.
    """
    x = np.clip(x, 1e-10, 1e10)
    log_x = np.log(x)
    power = np.exp(y * log_x)

    return power


class Clayton:
    def __init__(self):
        self.theta = 0
        self.bound = (-1, np.inf)

    def c(self, u: np.ndarray, v: np.ndarray):
        """
        return: np.array, the density of Clayton copula
        """
        return (1 + self.theta) * \
            mypower(u * v, -1 - self.theta) \
            * mypower(mypower(u, -self.theta) + 
                      mypower(v, -self.theta) - 1, 
                      -2 - 1 / self.theta)

    def h(self, u: np.ndarray, v: np.ndarray):
        """

        return: np.array, the h function or 
        partial derivative F(u|v) of Clayton copula
        since h function is basically a kind of conditional CDF,
        it should be between 0 and 1.

        """
        a = mypower(v, -self.theta - 1)
        b = mypower(u, -self.theta) \
                + mypower(v, -self.theta) - 1
        c = mypower(b, -1 - 1 / self.theta)
        result = a * c

        # todo check which theta value
        # will lead to nan value.
        if self.theta > 1000:
            result[np.isnan(result)] = 1
        else:
            result[np.isnan(result)] = 0
        
        result = np.clip(result, 0, 1)

        return result

    def inverse_h(self, w: np.ndarray, v: np.ndarray):

        """
        return: np.array, the inverse of h function,
        which is the conditional CDF of u given v.
        since the inverse of h function will lead to the x,
        which is uniform distributed, the value should be between 0 and 1.
        """

        a = w * mypower(v, self.theta + 1)
        b = mypower(a, -self.theta / (1 + self.theta))
        c = mypower(v, -self.theta)

        d = mypower(b + 1 - c, -1 / self.theta)

        # todo: to avoid the nan value     
        if self.theta > 1000:
            d[np.isnan(d)] = v[np.isnan(d)]
        else:
            d[np.isnan(d)] = w[np.isnan(d)]
        
        d = np.clip(d, 0, 1)
        return d


class Gaussian:
    def __init__(self):
        self.theta = 0.5
        self.bound = (-1+1e-6, 1-1e-6)

    def c(self, u, v):
        """
        return the density of Clayton copula
        """
        x1 = norm.ppf(u)
        x2 = norm.ppf(v)
        x1 = np.clip(x1, -1e10, 1e10)
        x2 = np.clip(x2, -1e10, 1e10)
        a = (self.theta ** 2) * (x1 ** 2 + x2 ** 2)\
        - 2 * self.theta * x1 * x2
        
        b = a / (2 * (1 - self.theta ** 2))
        return (1 / np.sqrt(1 - \
                            self.theta ** 2)) \
                            * np.exp(-b)

    def h(self, u, v):
        """
        return the h function
        """

        a = (norm.ppf(u) - self.theta * norm.ppf(v)) \
                / np.sqrt(1 - self.theta ** 2)
        

        return norm.cdf(a)

    def inverse_h(self, w, v):
        """
        return the inverse of h function,
        which is the conditional CDF of u given v.
        """

        a = norm.ppf(w) * np.sqrt(1 - self.theta ** 2)\
                + self.theta * norm.ppf(v)

        return norm.cdf(a)


