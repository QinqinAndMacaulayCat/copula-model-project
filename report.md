---
title: "Copula dependence and risk sensitivity of asset portfolio by Qinqin Huang, Yongyi Tang, Xiaohan Shen, and Pai Peng"
---


## 1 Introduction

In this report, we refer to the paper by Bruneau et al.<sup>[2](#Bruneau2019)</sup>
to estimate the risk sensitivity of financial assets through multivariate copula. The structure of the report is as follows. First, we analyze the data and build models to realize transforming data from ppf to cdf and the inverse process. Second, we introduce the canonical vine and simulate data from the canonical vine which have the same dependences with our input data as the parameters of the canonical vine are fitted based on the input data. Third, we calculate the Cross Conditional Value at Risk (CCVaR).


## 2 Modeling
### 2.1 Pair Copula

In this section, we basically refer to the models of the paper by Aas et al. <sup>[1](#Aas2009)</sup>.

#### 2.1.1 C-Vine

Considering a multivariate cumulative distribution function $F$ of $n$ random variables $\textbf{X}=({X_1, ..., X_n})$ with marginal cumulative distributions $F_1(x_1), ..., F_n(x_n)$, Skalar's Theorem states that there exists a unique n-dimensional copula $C$ to describte the joint distribution of these these marginals, which is defined as:

$$
F(x_1, x_2, ..., x_n) = C(F_1(x_1), F_2(x_2), ..., F_n(x_n)).
$$

Here, let $F_i(x_i)=u_i$, the copula $C$ can be written as:

$$
C(u_1, u_2, ..., u_n) = F(F_1^{-1}(u_1), F_2^{-1}(u_2), ..., F_n^{-1}(u_n)).
$$

If $F$ is absolutely continuous with strictly increasing and continuous marginal cdf $F_i$, the joint density function $f$ can be written as:

$$
f(x_1, x_2, ..., x_n) = c_{1:n}(F_1(x_1), F_2(x_2), ..., F_n(x_n)) \cdot \prod_{i=1}^{n} f_i(x_i).
$$ 

which is the product of the n-dimensional copula density $c_{1:n}(\cdot)$ of $C$ and the marginal densities $f_i(\cdot)$.

Building high-dimensional copulae is generally recognized as a challenging task. One of the most popular methods is the pair-copula construction (PCC) proposed by Aas et al. <sup>[1](#Aas2009)</sup>. The idea is to construct a high-dimensional copula by combining bivariate copulae. The basic principle behind PCC is that the density can be factorized as:

$$
f(x_1, x_2, ..., x_n) = f_n(x_n) \cdot f(x_{n-1}|x_n) \cdot f(x_{n-2}|x_{n-1}, x_n) \cdot ... \cdot f(x_1|x_2, ..., x_n). \, \tag{1}
$$

In a bivariate case, the density function is defined as:

$$
f(x_1, x_2) = c_{12} \{ F_1(x_1), F_2(x_2)\} \cdot f_1(x_1) \cdot f_2(x_2).
$$

For a conditional density, it follows that:

$$
f(x_1|x_2) = c_{12}(F_1(x_1), F_2(x_2)) \cdot f_1(x_1).
$$

For case with three random variables, the conditional density is given by:

$$
f(x_1|x_2, x_3) = c_{13|2}\{F(x_1|x_2), F(x_3|x_2)\}\cdot f(x_1|x_2) \\
$$

$$
= c_{13|2}\{F(x_1|x_2), F(x_3|x_2)\}\cdot c_{12}(F(x_1), F(x_2)) \cdot {f(x_1)}.
$$

where two pair-copulae are involved.

Based on the above, we can see each term in (1) can be decomposed into the appropriate pair-copula times a conditional marginal density, using the general formula:


$$
f(x| \textbf{v}) = c_{x v_j|\textbf{v}_{-j}}\{F(x|\textbf{v}_{-j}), F(v_j|\textbf{v}_{-j})\} \cdot f_{x}(\textbf{v}_{-j}). 
$$

Here $\textbf{v}$ is a vector of variables, $\textbf{v}_{-j}$ is the vector $\textbf{v}$ with the $j$-th element removed. 

The pair-copula construction involves marginal conditional distribution of the form $F(x|\textbf{v})$. Joe <sup>[4](#Joe1996)</sup> showed that, for every $j$ :

$$
F(x|\textbf{v}) = \frac{\partial C_{x v_j|\textbf{v}_{-j}}\{F(x|\textbf{v}_{-j}), F(v_j|\textbf{v}_{-j})\}}{\partial F(v_j|\textbf{v}_{-j})}.
$$

where $C_{ij|\textbf{v}}$ is a bivariate copula distribution function. d

For the special case where $v$ is a univariate, we have:

$$
F(x|v) = \frac{\partial C_{x v}(F(x), F(v))}{\partial F(v)}.
$$

We will use the function $h(x, v, \Theta)$ to represent this conditional distribution function when $x$ and $v$ are uniform, which is defined as:

$$
h(x, v, \Theta) = F(x|v) = \frac{\partial C_{x v}(F(x), F(v))}{\partial F(v)}, \\
$$


$$
\Theta - \text{the set of parameters of the joint distribution function}.
$$

where the second parameter of $h(\cdot)$ always corresponds to the conditioning variable and $\Theta$ denotes the set of parameters for the copula of the joint distribution function of $x$ and $v$. 


Furture, let $h^{-1}(u, v, \Theta)$ be the inverse of the h-function with respect to $u$, or the equivalently the inverse of the conditional distribution function.

For high-dimension distribution, there are significant number of possible pair-copular. To help organising them, Bedford and Cooke <sup>[5](#Cooke2001)</sup> have introduced a graphical model denoted as the regular vine. Here, we concentrate on the special case of regular vines - the canonical vine (C-vine), which gives a specific way of decomposing the density. The figure below cited from Czado and Naglar <sup>[3](#Czado)</sup> shows a C-vine with 5 variables. In a canonical vine tree all layers are stars: in every layer of the tree there is a single node, called the root, that is connecting all the others. In this figure, the root nodes are $1, (1, 4), (6, 4;1), (6,2;4,1), (5,2;6,4,1)$. [^1] Since all indices from previous root nodes are contained in the label of later root nodes, we can also specify the order by only referencing the index that enters in the next layer. For example the root node sequence in this figure can be written as $1, 4, 6, 2, 5, 3$.

[^1]: In the code, we call the root here as central node. 

![C-vine](reportfile/cvine.png)

\newpage

Based on the factorization discussed above, the n-dimensional density corresponding to a C-vine is given by:

$$
\prod_{k=1}^{n} f(x_k) \prod_{j=1}^{n-1} \prod_{i=1}^{n-j} c_{j, j+i|1, ..., j-1}\{F(x_j|x_1, ..., x_{j-1}), F(x_{j+i}|x_1, ..., x_{j+i-1})\}.
$$

Fitting a canonical vine might be advantageous when a particular variable is known to be a key variable that governs interaction in the data set. In such a situation one may decide to locate this variable at the root of the canonical vine, as we have done with variable in the figure. 

#### 2.1.2 Simulation from a pair-copula decomposed model

In this section we show the simulation algorithm for canonical vines which follows the method discussed in Aas <sup>[1](#Aas2019)</sup>.
We assume for simplicity that all the margins of the distribution are uniform. [^2]


[^2]: For variables with other marginal distributions, we transform the data to uniform marginals before fitting the vine copula.

To sample n dependent uniform[0, 1] variables, we first sample $w_1, ..., w_n$ independent uniform on [0, 1] and the variables $x_1, ..., x_n$ are generated by applying successive inverse cumulative distribution functions. We refer to the method mentioned by Cooke <sup>[6](#Cooke2007)</sup>. $w_1,...,w_n$ are values of $x_1, F(x_2|x_1), F(x_3|x_1, x_2), ... , F(x_n|x_1, x_2, ..., x_{n-1}$ respectively. And conditional distributions $F(x_n|x_1), F(x_n|x_1, x_2),...,F(x_n|x_1, x_2, ..., x_{n-1})$ can be found by conditionalizing copulae. Inverting the value of $w_n$ through $F(x_n|x_1), F(x_n|x_1, x_2), ... , F(x_n|x_1, x_2, ... , x_{n-1}$ gives $x_n$.
This process is illustrated in the Cooke's figure below:


![Staircase graph representation of canonical vine sampling procedure](reportfile/staircase.png)


\newpage


Sample $x_n$ as follows:

$$
x_n = F^{-1}_{x_n|x_1}(F^{-1}_{x_n|x_1,x_2}(...(F^{-1}_{x_n|x_1,...,x_{n-1}}(w_n))...)).
$$

As we mentioned before, the conditional distribution functions $F(x_i|x_1, ..., x_{i-1})$ can be computed by the h-function. Therefore, the algorithm is also <sup>[1](#Aas2009)</sup> for sampling from a canonical vine is as follows:


![simulation algorithm](reportfile/algo1.png)

\newpage

The outer loop runs over the variables to be sampled. This loop consists of two other for-loops. In the first, the ith varaible is sampled, while in the other, the conditional distribution functions needed for sampling the $(i+1)$th variable are updated. To compute these conditional distribution functions, we repeatedly use the h-function, with previously computed conditional distribution functions, $v_{i,j}=F(x_i|x_1, ..., x_{j-1})$, as the first two arguments.The last argument of the h-function is the parameter $\Theta_{j,i}$ of the corresponding copula density $c_{j,j+i}|1,...,j-1(\cdot, \cdot)$. 
The actually work flow for each loop is as follows(taking $i=n$ as example):

$$
h^{-1}(v_{n,1}, v_{n-1, n-1}, \Theta{n-1,1}) \\
$$

$$
=h^{-1}(w_i, F(x_{n-1}|x_1, ..., x_{n-2}), \Theta_{n-1,1}) \\
=F(x_n|x_1, ..., x_{i-2}).
$$

$$
h^{-1}{v_{n,1}, v_{n-2, n-2}, \Theta{n-2, 2}} \\
$$

$$
= h^{-1}(F(x_n|x_1, ..., x_{n-2}), F(x_{n-1}|x_1, ..., x_{n-3}), \Theta_{n-2, 2}) \\
$$

$$
= F(x_n|x_1, ..., x_{n-3}).
...
$$


$$
h^{-1}(v_{n,1}, v_{1, 1}, \Theta_{1, n-1}) \\
$$

$$
= h^{-1}(F(x_n|x_1), x_1, \Theta_{1, n-1}) = F(x_n).
$$



\newpage


#### 2.1.3 Estimation of the parameters

In this section we describe how the parameters of the canonical vine density are estimated. To simplify the process as mentioned before, we assumme that the marginals are uniform and the the time series is stationary and independent over time. This assumption is not limiting, as we can always preprocess the data through models such as ARIMA and GARCH to make the input of the canonical vine model stationary. 

We use the maximum likelihood method to estimate the parameters of the canonical vine. Since the actual margins are normally unknown in practice, what is being maximised is a pseudo-likelihood. 


The log-likelihood is given by:

$$
\sum_{j=1}^{n-1} \sum_{i=1}^{n-j} \sum_{t=1}^{T} \log c_{j, j+i|1, ..., j-1}\{F(x_{j,t}|x_{1,t}, ..., x_{j-1,t}), F(x_{i+j,t}|x_{1,t}, ..., x_{j-1,t})\}.
$$

For each copula in the above formula, there is at least one parameter to be determined. The algorithm for estimating the parameters is listed below in the figure. The ourter for-loop corresponds to the outer sum in the pseudo-likelihood. The inner for-loop corresponds to the sum over i. The innermost for-loop corresponds to the sum over the time series. Here, the element t of $textbf{v}_{j,i}$ is $v_{j, i, t} = F(x_{i, t}|x_{1,t},...,x_{j,t})$. $L(\textbf{x}, \textbf{v}, \Theta)$ is the log-likelihood of the chosen bivariate copula with parameters $\Theta$ and the data $\textbf{x}$ and $\textbf{v}$. That is,

$$
L(\textbf{x}, \textbf{v}, \Theta) = \sum_{t=1}^{T} \log c(x_t, v_t, \Theta),\\
c(u, v, \Theta) \text{is the density of the bivariate copula with parameters $\Theta$}.
$$

![estimation algorithm](reportfile/algolikelihood.png)

\newpage

Starting values of the parameters needed in the numerical maximization of the log-likelihood are determined as follows:

1. Estimate the parameters of the copulae in the first level of the vine tree from the original data.

2. Compute observations for level 2 using the copula parameters from level 1 and the h-function.

3. Estimate the parameters of the copulae in the second level of the vine tree from the observations computed in step 2.

4. Repeat steps 2 and 3 until the parameters of all copulae in the vine tree have been estimated.

#### 2.1.4 Copula selection

In the above content, we introduce the canonical vine copula, the calibration of the parameters, and the simulation of the data. However, we didn't specify which copula to use in the pair-copula decomposition. The choice of copula is crucial for the performance of the model. We only show the Gaussian copula and Clayton copula in the following content. However, the C-Vine structure can be easily extended to other copulae through getting copula functions and h-functions.


#### 2.1.4.1 Gaussian copula

The density of the bivariate Gaussian copula is given by:

$$
c(u, v, \theta) = \frac{1}{\sqrt(1-\theta^2)} exp \{ -\frac{{\theta}^2 (x_1^2 + x_2^2) - 2 \theta x_1 x_2}{2(1-\theta^2)} \}, -1 < \theta < 1.
$$

Here, $\theta$ is the correlation parameter, which is normally denoted as $\rho$. $x_1 = \Phi ^{-1}(u)$, $x_2 = \Phi^{-1}(v)$, and $\Phi$ is the standard normal distribution function.

The h-function is given by:

$$
h(u, v, \theta) = \Phi(\frac{\Phi^{-1}(u) - \theta \Phi^{-1}(v)}{\sqrt{1-\theta^2}}).
$$

Suppose the h-function is equal to $w$, then the inverse h-function is given by:

$$
h^{-1}(w, v, \theta) = \Phi\{ \Phi^{-1}(w) \sqrt{1-\theta^2} + \theta \Phi^{-1}(v) \}
$$


#### 2.1.4.2 Clayton copula

The density of Clayton copula is given by:

$$
c(u, v, \theta) = (1 + \theta)(u \cdot v)^{-\theta} - 1) \times (u^{-\theta} + v^{-\theta} - 1)^{-1/\theta - 2}, \theta \in [-1, \infty) \ 0. 
$$

Perfect dependence is obtained when $\theta \rightarrow \infty$.

For this copula the h-function is given by:

$$
h(u, v, \theta) = v^{-\theta-1}(u^{-\theta} + v^{-\theta} - 1){-1 - \theta}.
$$

Suppose the h-function is equal to $w$, then the inverse h-function is given by:

$$
h^{-1}(w, v, \theta) = \{(w \cdot v^{\theta+1})^{\frac{\theta}{\theta+1}} + 1-v^{-\theta}\}^{-1/\theta}.
$$

### 2.2 Gaussian Copula
A Gaussian Copula that the dependency structure between the multiple random variavles is Gaussian dependency. We introduce Gaussian Copula here to compare it with CVine.

#### 2.2.1 Mathematical Representation
Let $\Phi$ be the standard normal CDF and $\Phi_{\Sigma}$ is the CDF of a multivariate normal distribution with correlation matrix $\Sigma$. For random variables $(U_1, U_2, \ldots, U_d)$ with uniform marginals, the Gaussian Copula $(C)$ is defined as:

$$
C(u) = \Phi_{\Sigma}(\Phi^{-1}(u_1), \Phi^{-1}(u_2), \ldots, \Phi^{-1}(u_d))
$$

#### 2.2.2 Cholesky Decomposition
Cholesky decomposition is a matrix factorization technique used for symmetric, positive-definite matrix. It expresses a matrix as the product of a lower triangular matrix and its transpose.

Given a symmetric, positive-definite matrix $A$, the Cholesky decomposition finds a upper triangular matrix $U$ such that:

$$
A = LL^\top
$$

where:
- $L$ is a lower triangular matrix with real, positive diagonal entries.
- $L^\top$ is the transpose of $L$.

The algorithm is as follows:
1. For a given matrix $A$, calculate each element of $L$ using:
   $$
   L_{i,i} = \sqrt{A_{i,i} - \sum_{k=1}^{i-1} L_{i,k}^2}
   $$
   $$
   L_{i,j} = \frac{1}{L_{j,j}} \left( A_{i,j} - \sum_{k=1}^{j-1} L_{i,k} L_{j,k} \right), \quad \text{for } i > j
   $$
2. Fill the matrix $L$ row by row.

#### 2.2.3 Steps to Build a Gaussian Copula

1. Define the correlation matrix.

2. Use Cholesky decomposition to decompose the correlation matrix.

3. Generate independent standard normal random variables.

4. Apply the decomposed matrix on these random variables to get correlated random variables.

5. Map the random variables to uniform distributions by calculating their cumulative distribution function.


#### 2.2.4 Comparison between Gaussian Copula and CVine
In Gaussian Copula, the dependency is captured by a correlation matrix, while in CVne we models dependencies pairwisely. Here are some main difference between these two methods.

1. Gaussian Copula qssumes symmetric Gaussian dependency, which may not adequately represent tail dependencies. It underestimates the probability of extreme co-movements.

2. In CVine, dependency is modeled with separate copulas, allowing for different types of relationships between variables. CVine can capture asymmetric dependencies and tail dependencies more effectively.

3. Gaussian Copula only requires the Cholesky decomposition of the correlation matrix. So, the computation is flexible. However, CVine Copula More computationally intensive. It requires constructing and evaluating multiple pair-copulas and dependency trees.


### 2.3 CCVaR

The Cross Conditional Value at Risk (CCVaR) quantifies the expected return of an asset under the extreme conditions of a given risk factor. For an asset $R_i$ and a risk factor $X$, the CCVaR at confidence level $\alpha$ is defined as:

$$
CCVaR_\alpha(R_i \mid X; F_X) = \mathbb{E}[R_i \mid F_X(X) \leq \alpha],
$$

where:

$$
F_X(X): \text{ the cumulative distribution function (CDF) of the risk factor } X,
$$

$$
\alpha: \text{ the confidence level defining the extreme quantile (e.g., } \alpha = 0.05 \text{ for the worst 5 \%)}.
$$

\newpage

## 3 Implementation


### 3.1 Data Preprocessing
#### 3.1.1 Data Fetcher
1. **`__init__(self, tickers: list, start_date: datetime, end_date: datetime)`** - This method initializes the `DataFetcher` object. It takes in three arguments: 
   - `tickers`: a list of stock or index tickers,
   - `start_date`: the start date for fetching historical data,
   - `end_date`: the end date for the data fetching period. 

2. **`fetch_and_save_data(self)`** - It uses `yfinance` to download the adjusted closing prices for each ticker between the specified `start_date` and `end_date`. The downloaded data is then forward-filled for any missing values and saved to the CSV file. Finally, it computes the percentage change of the adjusted closing prices, dropping any rows with missing values.

3. **`plot_distribuion(self)`** - This method generates and displays histograms of the return distributions for each ticker in the `tickers` list.

Here, we choosed a list of tickers (tickers = ['^GSPC', '^DJI', '^TNX', '^IXIC', '^RUT']), representing the five financial instruments of S&P 500 Index, Dow Jones Industrial Average, CBOE 10-Year Treasury Note Yield, NASDAQ Composite Index and Russell 2000 Index. The start date of our data is set to January 1st, 2023, and the end date is set to November 1st, 2024. Using these parameters, we fetched historical market data of indices from Yahoo Finance. Then, we calculated the return of these indices and dropped invalid data as the input of our model.

#### 3.1.2 distribution
in this code, the `multivariate` class is designed to perform various multivariate statistical operations on a given dataset. the class includes methods for calculating empirical cumulative distribution functions (ecdf), empirical percent-point functions (ppf), extreme value correlation, and visualization of data through heatmaps and kernel density estimation (kde) plots. the main methods in the class are:

1. **`__init__(self, data)`** - this is the initialization method of the `multivariate` class. it takes a `data` argument. the method calculates and stores the covariance and correlation matrices of the data.

2. **`empircal_cdf(self)`** - this method computes the empirical cumulative distribution function (ecdf) for each column of the dataset. the ecdf is calculated by ranking the values in each column and dividing by the total number of data points. the result is stored in the `self.ecdf` attribute, and the method prints the rank of the data along with its length for verification. it returns the calculated ecdf values.

3. **`empircal_ppf(self, u)`** - this method calculates the empirical percent-point function (ppf) for a given set of quantiles (`u`). it iterates through each column in the data and computes the quantile value at the corresponding position in `u` for each column. the method returns an array of ppf values, which are the inverse of the ecdf.

4. **`extreme_value_correlation(df, percentile=95, direction="upper")`** - this static method computes the extreme value correlation for a dataset by analyzing the tail behavior of the data. it first calculates the threshold for each column at a given percentile. it then calculates the correlation of extreme values by checking how often two columns both exceed their respective thresholds (upper or lower). the result is returned as a correlation matrix showing the conditional probability of extreme values occurring together for each pair of columns.

5. **`heatmap(data, title)`** - this static method generates a heatmap visualization for the given data. it creates a heatmap from the data which can be used to visualize the relationships or correlations between different variables in the dataset.

6. **`plot_kde_comparison(df, title)`** - this method creates a pairplot to compare the kernel density estimates (kde) of the variables. the pairplot visualizes both scatter plots and kdes on the diagonal. this method is useful for understanding the pairwise relationships and distributions of variables in the dataset.

### 3.2 cvine

in this code, we use the class `cvine` to realize the canonical vine copula. basically, the class involves the following methods:

1. **build_tree()** - to build the tree structure of the canonical vine copula. this method will fill the class attribute `tree` with the tree structure. 

2. **fit()** - to fit the canonical vine copula to the data. this method will estimate the parameters of the copulae in the vine tree, which will call the method `get_likelihood()` to calculate the log-likelihood of the tree. here, we use `scipy.optimize.minimize` to maximize the log-likelihood.

3. **simulate()** - to simulate data from the canonical vine copula. this method will simulate data from the fitted vine copula. in this algorithm, we generate independent uniform random variables and then use the algorithm mentioned in section 2 to generate dependent uniform random variables.

### 3.2 gaussian copula
in this code, we use the class `gaussiancopula` to implement a gaussian copula for modeling dependencies between multiple variables. the class involves several key methods, each performing distinct tasks in the copula modeling process:

1. **estimate_paras()** - this method estimates the parameters (mean and standard deviation) for each variable in the dataset. it calculates the mean (`miu`) and standard deviation (`sigma`) for each variable in the dataset and stores these parameters in the `parameter_dict` attribute. these parameters are essential for understanding the marginal distributions of the individual variables before applying the copula.

2. **estimate_corr()** - this method estimates the correlation matrix from the dataset. it first centers the data by subtracting the mean of each variable and then computes the covariance matrix. the covariance matrix is normalized by dividing by the product of the standard deviations of the variables to obtain the correlation matrix. this matrix captures the dependencies between the variables, which will later be used to introduce correlation in the simulated data.

3. **generate_samples(n_samples)** - this method generates samples from the fitted gaussian copula. it first creates independent random variables using the `generate_normal_bm` function, which generates standard normal random variables using the box-muller method. then, it applies the cholesky decomposition to the correlation matrix to introduce the dependency structure between the variables. the uncorrelated normal variables are multiplied by the cholesky factor, resulting in correlated normal variables. these are then transformed into uniform random variables using the cumulative distribution function (cdf) of the normal distribution. finally, the uniform random variables are mapped back to the marginal distributions using the inverse cumulative distribution (quantile function), generating correlated sample returns from the copula.


### 3.3 ccvar

in this section, we describe the code implementation of ccvar using python. the implementation is encapsulated in the `ccvar` class, which contains methods to calculate ccvar for single asset-factor pairs and generate a ccvar matrix for all assets and factors.

1. **initialization**: the `__init__` method initializes the ccvar model by taking the following inputs:

- `data`: asset return matrix ($t \times n$).
- `factors`: risk factor matrix ($t \times f$).
- `alpha`: confidence level for defining extreme conditions.

2. **data transformation**: the `_transform_to_uniform` method transforms raw data to the uniform space $[0, 1]$ using the empirical cumulative distribution function (cdf).

3. **extreme event identification**: the `_get_extreme_indices` method identifies indices corresponding to extreme events, where the risk factor falls below the $\alpha$-quantile.

4. **single ccvar calculation**: the `calculate_ccvar` method computes ccvar for a single asset with respect to a specific risk factor.

5. **ccvar matrix calculation**: the `calculate_all_ccvar` method generates a matrix of ccvar values for all assets and risk factors.

6. **result summarization**: the `summarize_results` method outputs the ccvar matrix with labels for assets and factors.

\newpage


### 3.3 results

#### 3.3.1 cvine and multivariate gaussian copula
we test the `cvine` and `gaussiancopula`based on the return data of the assets. 

we first show the correlation matrix of the returns of our initial data and the correlation of the returns of the simulated data from cvine (gaussian copula and clayton copula) and multivariate gaussian copula. the results are shown below:

![correlation matrix of the returns of the initial data](result/correlation_matrix_of_initial_data.png)

\newpage

![correlation matrix of the returns of the simulated data from gaussian copula](result/correlation_matrix_of_gaussian_copula.png)

\newpage

![correlation matrix of the returns of the simulated data from clayton copula](result/correlation_matrix_of_clayton_copula.png)


\newpage
![correlation matrix of the returns of the simulated data from clayton copula](result/correlation_matrix_of_gaussianm.png)


\newpage

basically, the correlation matrix of the returns of the simulated data from the copulae is similar to the correlation matrix of the returns of the initial data. however, we can see that the level of pearson correlation is different. 

then we show the scatter plot of the returns of the initial data and the scatter plot of the returns of the simulated data from cvine and gaussian copula. the results are shown below: 

![scatter plot of the returns of the initial data](result/compare_cdfs_of_initial_data.png)

\newpage

![scatter plot of the returns of the simulated data from gaussian copula](result/compare_cdfs_of_gaussian_copula.png)

\newpage

![scatter plot of the returns of the simulated data from clayton copula](result/compare_cdfs_of_clayton_copula.png)

\newpage

![scatter plot of the returns of the simulated data from clayton copula](result/compare_cdfs_of_gaussian_copula.png)

\newpage

we can see that the scatter plot of the returns of the simulated data from the copulae is basically similar to the scatter plot of the returns of the initial data, which means the copulae can capture the dependence structure of the data. however, the correlation of the simulated results may perform differently from the initial data when the assets don't have clear correlation structure.


finally, we test the extreme dependence between the returns of the assets as the clayton copula should have reflected the lower tail dependence. we simply calculate a matrix to evaluate the level of extreme dependence between the returns of the assets. for the value in i-th row and j-th column, it is the probability that the return of the j-th asset is below the 5% quantile given the return of the i-th asset is below the 5% quantile. the results are shown below:

![extreme dependence between the returns of the assets](result/lower_correlation.png)

\newpage

in cvine, the average value of the clayton copula is higher than the gaussian copula, which means the clayton copula has a higher level of extreme dependence between the returns of the assets.

in multivariate gaussian copula, we get a even higher correlation. it may mean that the multivariate gaussian copula overestimates linear dependency between variables.

our results shows that cvine are more applicable in more complex dependent relationships.


#### 3.3.2 ccvar

\newpage

## 4 conclusion

the canonical vine copula is a powerful tool for modeling the dependence structure of multivariate data. in this report, we have introduced the canonical vine copula and its application in estimating the risk sensitivity of asset portfolios. we have implemented the canonical vine copula in python and demonstrated its use in simulating data and estimating the parameters of the copulae. besides, we compared our results of cvine and multivariate gaussian copula and find that cvine is more useful in depicting non-linear relationships. we have also implemented the cross conditional value at risk (ccvar) to quantify the expected return of an asset under extreme conditions of a given risk factor. the ccvar provides a useful measure of the risk sensitivity of asset portfolios to different risk factors. further research can explore the application of the canonical vine copula in asset portfolio optimization and risk management. the drawbacks of our implementation include that we have not compare the results of different copulae and simply assume returns follow gaussian copula. 

\newpage

## references

<a id="aas2009"></a> [1] aas, k., czado, c., frigessi, a., and bakken, h. (2009). pair-copula constructions of multiple dependence. insurance: mathematics and economics, 44(2), 182-198. 

<a id="bruneau2019"></a> [2] catherine bruneau, alexis flageollet, and zhun peng. (2019). vine copula based modeling.  

<a id="czado"></a> [3] claudia czado and thomas naglar. (2021). vine copula based modeling.

<a id="joe1996"></a> [4] joe, h., 1996. families of m-variate distributions with given marginals and bivariate dependence parameters.

<a id="cooke2001"></a> [5] bedford, t., cooke, r.m., 2001b. probability density decomposition for conditionally dependent random variables modeled by vines. annals of mathematics and artificial intelligence 32, 245–268.

<a id="cooke2007"></a> [6] d. kurowicka, r.m. cooke, sampling algorithms for generating joint uniform distributions using the vine-copula method, computational statistics & data analysis, volume 51, issue 6, 2007, pages 2889-2906, issn 0167-9473, https://doi.org/10.1016/j.csda.2006.11.043.

## appendix

### a code

#### a.1 cvine

```python

import numpy as np
from scipy.optimize import minimize
from copula import clayton, gaussian

class cvine(object):
    layer = {"root": [], 
             # list of root nodes.
             #ex. in f(u1, u2|v), v is the root node 
             "parentnode": {},
             # index of nodes in last level. 
             # ex. {1: (1,2)} means the node 1 in this tree level 
             # got from the node pair (1,2) in last level
             "node": [], 
             # index of the nodes. 
             # from 0 to l. this is not the initial index.
             "pair": [],
             # list of node pairs in the tree,
             # ex. in f(u1, u2|v), (u1, u2) is a node pair. 
             # here the node pair is the index of nodes in the root,
             #which is different from the "node". 
             "level": 0, 
             # level of the tree (k). 
             # 0-root, 1-1st level, 2-2nd level, ...
             "nodenum": 0, 
             # number of the nodes in this tree (l). 
             # equal to n - k 
             "edgenum": 0, 
             # number of the edges in this tree.
             # equal to l as our node number is
             # the actual number minus1.
             "v": none,  
             # h functions in this level. 
             #v[:, j] is the h function of node j.
             }    

    tree = {"thetamatrix": none,
            # copula parameter matrix in this level.
            # it is a upper matrix. thetamatrix[i, j] is 
            # the copula parameter in level j 
            # between node 1 and node i+1. 
            "structure": {}, 
            # the tree structure in this level.
            # the key is the node index, 
            # the value is layer.
            "depth": 0, 
            # the depth of the tree, 
            # 0 means only has root. 
            }

    def __init__(self, u, copulatype="clayton"):
        
        """
        u: np.array, data matrix. 
        follows uniform distribution

        """
        self.u = u
        self.t = u.shape[0]
        self.variable_num = u.shape[1] - 1 
        # to make the structure more clear, 
        # all the variables are indexed from 0. 
        # therefore, when the variable_num is n,
        # we actually have n+1 variables x0, x1, ..., xn.
        if copulatype == "clayton":
            self.copula = clayton()
        elif copulatype == "gaussian":
            self.copula = gaussian()

        else:
            raise valueerror("the copula type\
                             is not supported.")

        self.max_depth = self.variable_num  
        # todo: the max_depth is not implemented yet.
        
    def build_tree(self):
        """
        build the tree structure. 
        """
        self.build_root()
        while self.tree["depth"] < self.max_depth:
            self.build_kth_tree()

    def build_root(self):
        """
        build the root of the tree. the root is basically 
        """
        
        layer = self.layer.copy()
        layer["level"] = 0
        layer["v"] = self.u.copy() 
        # the f(x|v) in the first layer 
        # is the empirical cdf of x. 
        layer["nodenum"] = self.variable_num
        layer["edgenum"] = self.variable_num 
        layer["node"] = list(range(0, layer["nodenum"] + 1))
        self.tree["structure"][0] = layer

        
    def build_kth_tree(self):
        """
        build the kth tree. 
        """

        if self.tree["depth"] >= self.variable_num:
            print("the tree depth is already the maximum.")

        last_layer =\
         self.tree["structure"][self.tree["depth"]]

        layer = self.layer.copy()
        layer["level"] = \
         last_layer["level"] + 1
        layer["nodenum"] =\
         last_layer["nodenum"] - 1
        layer["edgenum"] = layer["nodenum"]
        layer["node"] = \
        list(range(0, layer["nodenum"]+ 1))
        (layer["pair"], layer["node"], \
         layer["parentnode"], layer["root"]) = \
        self.pair_nodes(last_layer)
        self.tree["structure"][layer["level"]] = layer
        self.tree["depth"] = self.tree["depth"] + 1
    
    def pair_nodes(self, last_layer):
        """
        pair the nodes in this layer. 
        here we use the first node in each level 
        as the new central node and combine it 
        with the root in last level to get the new root. 
        this process is same as the process we show in the report.
        """


        nodes = range(0, last_layer["nodenum"] + 1)

        if last_layer["level"] == 0: 
            # the second layer is not conditional copula,
            # so we just combine the center node with 
            # neighor nodes without any condition.
            pair_left = last_layer["node"][0]
            pairs = tuple(zip(last_layer["edgenum"] * \
                              [pair_left], 
                              last_layer["node"][1:]))
            parentnodes = dict(zip(nodes, pairs))
            dependent = np.empty(last_layer["nodenum"] + 1)
            return (pairs, 
                   nodes, 
                   parentnodes, 
                   [])
        else:
            pairs = []
            parentnodes = {}
            last_pairs = last_layer["pair"]
            
            common_node = last_pairs[0][0] 
            # set the first node as center node in each layer.

            new_root = \
            last_layer["root"] + [common_node]
            pair_left = last_pairs[0][1] 
            # the right element in the center pair will be
            # the left element in pairs in this layer. 
            for i in range(1, last_layer["nodenum"] + 1):
                pairs.append(tuple((pair_left, last_pairs[i][1])))
                parentnodes[i-1] = (0, i)  
                # the i-1th node in this layer is from 
                # the pair (0, i) in last layer. ex.
                # the first layer in the second layer is
                # from the node pair (0, 1) in the first layer.   
        return (pairs, 
                nodes,
                parentnodes, 
                new_root)


    def fit(self):
        """
        fit the vine tree model by maximizing 
        the likelihood of the whole tree.

        """
        paramnum =\
        sum([self.tree["structure"][layer]["edgenum"] \
             for layer in range(0, self.tree["depth"])])

        thetaparams = np.ones(paramnum) * 0.5
        bounds = [self.copula.bound] * paramnum
        result = minimize(self.get_likelihood, \
                          thetaparams, bounds=bounds)
        thetamatrix = np.zeros((self.tree["depth"], \
                    self.tree["structure"][0]["edgenum"]))
        n = 0
        print("result", result)
        for i in range(0, self.tree["depth"]):
            for j in range(0, 
                self.tree["structure"][i]["edgenum"]):
                thetamatrix[i, j] = result.x[n]
                n += 1

        self.tree["thetamatrix"] = thetamatrix


    def fit2(self):
        
        """
        fit the parameters through maximizing
        the likelihood in each layer.
        """
        self.tree["thetamatrix"] =\
                np.zeros((self.tree["depth"], 
                self.tree["structure"][0]["edgenum"]))

        for i in range(1, self.tree["depth"]+1):
            last_layer =\
                    self.tree["structure"][i-1]
            layertheta = \
                np.ones(last_layer["edgenum"]) * 0.5
            bounds = \
                [self.copula.bound] * \
                last_layer["edgenum"]
            result = \
                minimize(self.get_layer_likelihood, 
                         layertheta, args=(last_layer, ), 
                         bounds=bounds)
            self.tree["thetamatrix"][i-1, \
                     :last_layer["edgenum"]] = result.x    

            self.tree["structure"][i]["v"] = \
                    self.get_layer_h(result.x, last_layer)


    def simulate(self, n):
        """
        simulate the data from the vine tree model
        param n: int, the number of the data 
            to be simulated for each variable.

        """
        if self.tree["thetamatrix"] is none:
            print("please fit the model first.")
            return none
        
        else:

            w = np.random.uniform(0, 1, \
                    n * (self.variable_num + 1))
            v = np.empty((n,
                          self.variable_num+1, 
                          self.variable_num+1))
            w = w.reshape((n, 
                           self.variable_num + 1))
            u = np.empty((n, 
                          self.variable_num + 1))
            u[:, 0] = w[:, 0]
            v[:, 0, 0] = w[:, 0] 
            for i in range(1, 
                           self.tree["depth"] + 1):
                v[:, 0, i] = w[:, i]
                k = i - 1
                while k >= 0:
                    self.copula.theta = \
                    self.tree["thetamatrix"][k, i-k-1]
                    v[:, 0, i] = \
                    self.copula.inverse_h(v[:, 0, i], 
                                          v[:, k, k])
                    k -= 1

                u[:, i] = v[:, 0, i]

                for j in range(0, i): 
                    self.copula.theta =\
                    self.tree["thetamatrix"][j, i-j-1]
                    v[:, j + 1, i] =\
                    self.copula.h(v[:, j, i], v[:, j, j])
            
            return u
        

    def get_likelihood(self, thetaparams):
        """get the likelihood of the vine tree model"""
        
        total_likelihood = 0
        left = 0 
        right = 0
        # ignore the root layer
        for k in range(1, 
                self.tree["depth"] + 1):            
            # each layer' c function is determined
            # by the last layer's and this layer's theta. 
            #number of theta in each layer is 
            # equal to the number of nodes in this layer.
            last_layer = self.tree["structure"][k-1]

            left = right
            right = right + last_layer["edgenum"] 
            layertheta = thetaparams[left:right]
            total_likelihood += \
                self.get_layer_likelihood(layertheta,
                                        last_layer)
            
            self.tree["structure"][k]["v"] = \
                    self.get_layer_h(layertheta, 
                                     last_layer)
        
        return total_likelihood

    def get_layer_likelihood(self, 
                             thetaparams,
                             last_layer):
        """get the likelihood of the layer"""
        likelihood = 0
    
        for i in range(1, 
                       last_layer["nodenum"]+1): 
            # totally l copula functions
            
            self.copula.theta = thetaparams[i-1]
            c = self.copula.c(last_layer["v"][:, 0], 
                              last_layer["v"][:, i])       
            c = np.clip(c, 1e-10, np.inf) 
            # to avoid the log(0) problem
            likelihood += np.sum(np.log(c))
        
        return -likelihood

    def get_layer_h(self, thetaparams, 
                    last_layer):
        """get the h function of the layer"""
        v = np.empty((self.t, last_layer["nodenum"])) 
        # the total nodes of this layer is 
        # the number of nodes in last layer minus 1, 
        # which is equal to the edges in last layer.
        
        for i in range(1, 
                       last_layer["nodenum"]+1):
            self.copula.theta = thetaparams[i-1]
            v[:, i-1] = \
                    self.copula.h(last_layer["v"][:, i],
                                last_layer["v"][:, 0])
        
        return v

```

\newpage

#### a.2 copula

```python

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


class clayton:
    def __init__(self):
        self.theta = 0
        self.bound = (-1, np.inf)

    def c(self, u: np.ndarray, v: np.ndarray):
        """
        return: np.array, the density of clayton copula
        """
        return (1 + self.theta) * \
            mypower(u * v, -1 - self.theta) \
            * mypower(mypower(u, -self.theta) + 
                      mypower(v, -self.theta) - 1, 
                      -2 - 1 / self.theta)

    def h(self, u: np.ndarray, v: np.ndarray):
        """

        return: np.array, the h function or 
        partial derivative f(u|v) of clayton copula
        since h function is basically a kind of conditional cdf,
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
        which is the conditional cdf of u given v.
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


class gaussian:
    def __init__(self):
        self.theta = 0.5
        self.bound = (-1+1e-6, 1-1e-6)

    def c(self, u, v):
        """
        return the density of clayton copula
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
        which is the conditional cdf of u given v.
        """

        a = norm.ppf(w) * np.sqrt(1 - self.theta ** 2)\
                + self.theta * norm.ppf(v)

        return norm.cdf(a)

```

\newpage

#### a.3 gaussian copula

```python

import numpy as np
import math
from scipy.stats import norm


def cal_determinant(matrix):
    if matrix.shape[0] == 1:

        return matrix[0, 0]

    elif matrix.shape[0] == 2:

        return matrix[0, 0] * matrix[1, 1] -
        matrix[0, 1] * matrix[1, 0]

    else:

        determinant = 0

        for i in range(matrix.shape[0]):
            sub_matrix = np.hstack((matrix[1:, :i],
            matrix[1:, i + 1:]))
            determinant_sub = cal_determinant(sub_matrix)
            determinant += (-1) ** i * matrix[0, i] * 
            determinant_sub

        return determinant


def symmetric_matrix(matrix, if_print=true):
    is_symmetric = true

    if matrix.shape[0] != matrix.shape[1]:
        print("this is not a matrix.")
        is_symmetric = false

    for i in range(matrix.shape[0]):
        for j in range(i):
            if matrix[i, j] != matrix[j, i]:
                is_symmetric = false
    if if_print:
        if is_symmetric:
            print("the matrix is a symmetric matrix.")

        else:
            print("the matrix is not symmetric matrix.")

    return is_symmetric


def positive_definite_matrix(matrix):
    is_pd = true

    if not symmetric_matrix(matrix):
        print("the matrix is not positive definite matrix.")
        return false

    for i in range(matrix.shape[0]):
        if cal_determinant(matrix[:i, :i]) <= 0:
            is_pd = false


    return is_pd


def cholesky_decomposition(matrix):
    n = matrix.shape[0]
    u = np.zeros((n, n))

    for i in range(n):
        # sum_square = sum(d[k, i] ** 2 for k in range(i))
        sum_square = np.dot(u[:i, i], u[:i, i])
        u[i, i] = np.sqrt(matrix[i, i] - 
        sum_square)

        for j in range(i + 1, n):
            # sum_ = sum(d[k, i] * d[k, j] for k in range(j))
            sum_ = np.dot(u[:j, i], u[:j, j])
            u[i, j] = (matrix[i, j] - sum_) / u[i, i]

    return u


def generate_normal_bm(miu, sigma, n):
    # generate d ~ exp(1 / 2)
    d = - 2 * np.log(np.random.uniform(0,
    1, int(n / 2)))

    # generate \theta ~ unif(\theta, 2\pi)
    theta = 2 * math.pi * np.random.uniform(0,
    1, int(n / 2))

    # generate x, y ~ normal(miu, sigma)
    x = np.sqrt(d) * np.cos(theta) *
        sigma + miu
    y = np.sqrt(d) * np.sin(theta) * 
    sigma + miu
    normal_random_variables = np.hstack((x, y))

    return normal_random_variables


class gaussiancopula(object):
    def __init__(self, data, tickers):
        self.data = np.array(data)
        self.tickers = tickers
        self.n_index = self.data.shape[1]
        self.parameter_dict = {}
        self.corr = np.array([])

    def estimate_paras(self):
        return_data = self.data
        self.parameter_dict = {}
        for ticker in self.tickers:
            miu = np.mean(return_data)
            sigma = np.std(return_data, ddof=0)
            self.parameter_dict[ticker] = [miu, sigma]

    def estimate_corr(self):
        # estimate the covariance matrix
        mean = np.mean(self.data, axis=0)
        demeaned_data = self.data - mean
        covariance = (demeaned_data.t @ demeaned_data) / (self.n_index - 1)
        std = np.sqrt(np.diag(covariance))
        self.corr = covariance / np.outer(std, std)

    def generate_samples(self, n_samples):
        # generate random variables
        random_normal = generate_normal_bm(0, 1,
        n_samples * self.n_index).reshape(
                                n_samples, 
                                self.n_index)
        u_matrix = cholesky_decomposition(self.corr)
        correlated_normal = random_normal @ u_matrix
        # convert it into u[0, 1]
        u = norm.cdf(correlated_normal)
        # map to marginal distributions
        sample_returns = []
        for i in range(self.data.shape[1]):
            ppf_value = np.quantile(self.data[:, i], 
                                u[:, i])
            sample_returns.append(ppf_value)

        sample_returns = np.array(sample_returns).t

        return u, sample_returns

```


