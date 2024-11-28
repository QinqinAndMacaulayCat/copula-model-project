import numpy as np
import math
from scipy.stats import norm


def cal_determinant(matrix):
    if matrix.shape[0] == 1:

        return matrix[0, 0]

    elif matrix.shape[0] == 2:

        return matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]

    else:

        determinant = 0

        for i in range(matrix.shape[0]):
            sub_matrix = np.hstack((matrix[1:, :i], matrix[1:, i + 1:]))
            determinant_sub = cal_determinant(sub_matrix)
            determinant += (-1) ** i * matrix[0, i] * determinant_sub

        return determinant


def symmetric_matrix(matrix, if_print=True):
    is_symmetric = True

    if matrix.shape[0] != matrix.shape[1]:
        print("This is not a matrix.")
        is_symmetric = False

    for i in range(matrix.shape[0]):
        for j in range(i):
            if matrix[i, j] != matrix[j, i]:
                is_symmetric = False
    if if_print:
        if is_symmetric:
            print("The matrix is a symmetric matrix.")

        else:
            print("The matrix is not symmetric matrix.")

    return is_symmetric


def positive_definite_matrix(matrix):
    is_pd = True

    if not symmetric_matrix(matrix):
        print("The matrix is not positive definite matrix.")
        return False

    for i in range(matrix.shape[0]):
        if cal_determinant(matrix[:i, :i]) <= 0:
            is_pd = False


    return is_pd


def cholesky_decomposition(matrix):
    n = matrix.shape[0]
    U = np.zeros((n, n))

    for i in range(n):
        # sum_square = sum(D[k, i] ** 2 for k in range(i))
        sum_square = np.dot(U[:i, i], U[:i, i])
        U[i, i] = np.sqrt(matrix[i, i] - sum_square)

        for j in range(i + 1, n):
            # sum_ = sum(D[k, i] * D[k, j] for k in range(j))
            sum_ = np.dot(U[:j, i], U[:j, j])
            U[i, j] = (matrix[i, j] - sum_) / U[i, i]

    return U


def generate_normal_bm(miu, sigma, n):
    # generate D ~ Exp(1 / 2)
    d = - 2 * np.log(np.random.uniform(0, 1, int(n / 2)))

    # generate Θ ~ Unif(0, 2Π)
    theta = 2 * math.pi * np.random.uniform(0, 1, int(n / 2))

    # generate X, Y ~ Normal(miu, sigma)
    X = np.sqrt(d) * np.cos(theta) * sigma + miu
    Y = np.sqrt(d) * np.sin(theta) * sigma + miu
    normal_random_variables = np.hstack((X, Y))

    return normal_random_variables


# def normal_cdf(x):
#     return (1 + math.erf(x / math.sqrt(2))) / 2
#
#
# def normal_ppf(p, tol=1e-6):
#     a, b = -10, 10
#     mid = (a + b) / 2
#     while b - a > tol:
#         if normal_cdf(mid) < p:
#             a = mid
#         else:
#             b = mid
#         mid = (a + b) / 2
#
#     return mid

class GaussianCopula(object):
    def __init__(self, data, tickers):
        self.data = np.array(data)
        self.tickers = tickers
        self.n_index = self.data.shape[1]
        self.parameter_dict = {}
        self.corr = np.array([])

    def estimate_paras(self):
        return_data = self.data
        self.parameter_dict = {}
        # assume all the returns satisfies gaussian distribution, estimate the parameters sigma and miu
        for ticker in self.tickers:
            miu = np.mean(return_data)
            sigma = np.std(return_data, ddof=0)
            self.parameter_dict[ticker] = [miu, sigma]

    def estimate_corr(self):
        # assume the returns satisfies Multivariate Gaussian Distribution, estimate the covariance matrix
        mean = np.mean(self.data, axis=0)
        demeaned_data = self.data - mean
        covariance = (demeaned_data.T @ demeaned_data) / (self.n_index - 1)
        std = np.sqrt(np.diag(covariance))
        self.corr = covariance / np.outer(std, std)

    def generate_samples(self, n_samples):
        # generate random variables from the estimated distribution
        random_normal = generate_normal_bm(0, 1, n_samples * self.n_index).reshape(n_samples, self.n_index)
        U_matrix = cholesky_decomposition(self.corr)
        correlated_normal = random_normal @ U_matrix
        # convert it into U[0, 1]
        U = norm.cdf(correlated_normal)
        # map to marginal distributions
        sample_returns = []
        i = 0
        for ticker in self.tickers:
            miu, sigma = self.parameter_dict[ticker]
            returns_single_index = norm.ppf(U[:, i]) * sigma + miu  # Asset A returns
            sample_returns.append(returns_single_index)

        sample_returns = np.array(sample_returns).T

        return U, sample_returns

# tickers = ['^GSPC', '^DJI', '^TNX', '^IXIC', '^RUT']
# start_date = '2023-01-01'
# end_date = '2024-11-01'
#
# dft = DataFetcher(tickers=tickers, start_date=start_date, end_date=end_date)
# dft.fetch_and_save_data()
# dft.plot_distribuion()
