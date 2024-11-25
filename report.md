---
title: "Copula-based simulation of multivariate time series by Qinqin Huang, Yongyi Tang, Xiaohan Shen, and Pai Peng"
---


## 1 Introduction


## 2 Modeling
### 2.1 Copulae
#### 2.1.1 C-Vine
Considering a multivariate cumulated distribution function $F$ of $n$ random variables $\textbf{X}={X_1, ..., X_n}$ with marginals $F_1(x_1), ..., F_n(x_n)$, Skalar's Theorem states that there exists a unique n-dimensional copula $C$ to describte the joint distribution of these these variables. If $F$ is absolutely continuous with strictly increasing and continuous marginal cdf $F_i$, the joint density function $f$ can be written as:

$$
f(x_1, x_2, ..., x_n) = c_{1:n}(F_1(x_1), F_2(x_2), ..., F_n(x_n)) \cdot \prod_{i=1}^{n} f_i(x_i)
$$

which is the product of the n-dimensional copula density $c_{1:n}(\cdot)$ of $C$ and the marginal densities $f_i(\cdot)$.

Building high-dimensional copulae is generally recognized as a challenging task. One of the most popular methods is the pair-copula construction (PCC) proposed by Aas et al. [1]<sup>[1](#Aas2009)</sup>. The idea is to construct a high-dimensional copula by combining bivariate copulae. The basic principle behind PCC is that the density can be factorized as:

$$
f(x_1, x_2, ..., x_n) = f_n(x_n) \cdot f(x_{n-1}|x_n) \cdot f(x_{n-2}|x_{n-1}, x_n) \cdot ... \cdot f(x_1|x_2, ..., x_n) \, \tag{1}
$$

Each term in (1) can be decomposed into the appropriate pair-copula times a conditional marginal density, using the general formula:

$$
f(x| \textbf{v}) = c_{x v_j|\textbf{v}_{-j}}\{F(x|\textbf{v}_{-j}), F(v_j|\textbf{v}_{-j})\} \cdot f_{x}(\textbf{v}_{-j}) \, \tag{2}
$$
Here $\textbf{v}$ is a vector of variables, $\textbf{v}_{-j}$ is the vector $\textbf{v}$ with the $j$-th element removed. 

The pair-copula construction involves marginal conditional distribution of the form $F(x|\textbf{v})$. For every $j$ :

$$
F(x|\textbf{v}) = \frac{\partial C_{x v_j|\textbf{v}_{-j}}\{F(x|\textbf{v}_{-j}), F(v_j|\textbf{v}_{-j})\}}{\partial F(v_j|\textbf{v}_{-j})} 
$$

For the special case where $v$ is a univariate, we have:

$$
F(x|v) = \frac{\partial C_{x v}(F(x), F(v))}{\partial F(v)}
$$

We will use the function $h(x, v, \Theta)$ to represent this conditional distribution function when $x$ and $v$ are uniform:

$$
h(x, v, \Theta) = F(x|v) = \frac{\partial C_{x v}(F(x), F(v))}{\partial F(v)}, \Theta - the set of parameters of the joint distribution function.
$$

Furture, let $h^{-1}(u, v, \Theta)$ be the inverse of the h-function with respect to $u$, or the equivalently the inverse of the conditional distribution function.

For high-dimension distribution, there are significant number of possible pair-copular. To help organising them, Bedford and Cooke have introduced a graphical model denoted as the regular vine. Here, we concentrate on the special case of regular vines - the canonical vine (C-vine), which gives a specific way of decomposing the density. The figure below shows a C-vine with 5 variables. 


The n-dimensional density corresponding to a C-vine is given by:

$$
\prod_{k=1}^{n} f(x_k) \prod_{j=1}{n-1} \prod_{i=1}^{n-j} c_{i, i+j|1, ..., j-1}\{F(x_j|x_1, ..., x_{j-1}), F(x_{i+j}|x_1, ..., x_{i+j-1})\}
$$

Fitting a canonical vine might be advantageous when a particular variable is known to be a key variable that governs interaction in the data set. In such a situation one may decide to locate this variable at the root of the canonical vine, as we have done with variable in the figure. 


#### 2.1.2 Conditional independence


#### 2.1.3 Bivariate copulae

#### 2.1.4 Simulation from a pair-copula decomposed model

### 2.2 Distribution of the data


### 2.3 CCVaR


## 3 Implementation
### 3.1 Data

### 3.2 Code

### 3.3 Results


## 4 Conclusion




## References

<a id="Aas2009"></a> [1] Aas, K., Czado, C., Frigessi, A., and Bakken, H. (2009). Pair-copula constructions of multiple dependence. Insurance: Mathematics and Economics, 44(2), 182-198.
<a id="Bruneau2019"></a> [2] Catherine Bruneau, Alexis Flageollet, and Zhun Peng. (2019). VineCopula: Statistical Inference of Vine Copulas. R package version 2.0.0. https://CRAN.R-project.org/package=VineCopula

