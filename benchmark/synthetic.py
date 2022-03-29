import numpy as np
import pandas as pd
from scipy import stats

n = 100
p = 50
beta = np.random.uniform(size=p).reshape(-1, 1)*5
alpha = np.random.uniform() + 1


# https://onlinelibrary.wiley.com/doi/full/10.1002/sim.9136

def generating_data(n, p):
    pi_k = np.random.uniform(0.2, 0.8, size=p)
    mu_k = stats.norm.ppf(pi_k)
    sigma_ = np.random.normal(size=p * p).reshape(p, p)
    s = np.dot(sigma_.T, sigma_)
    d = s.diagonal()
    sigma = s / (d.reshape(-1, 1) * d.reshape(1, -1))
    mg = stats.multivariate_normal(mu_k, sigma, allow_singular=True)
    x_tilde = mg.rvs(size=n) > 0
    return x_tilde


def generate_time_of_event(x, alpha, beta):
    m = np.exp(alpha + np.dot(x, beta)).ravel()
    return np.random.weibull(alpha, size=len(x)) * m


def generate_time_censoring(x, alpha, beta):
    m = np.mean(np.exp(alpha + np.dot(x, beta)).ravel())
    return np.random.weibull(alpha, size=len(x)) * m


def generate_left_truncation(x, mu, sigma):
    return np.random.lognormal(mu, sigma, size=len(x))


def density_function(x, t, alpha, beta):
    t = t.reshape(1, -1)
    m = np.exp(alpha + np.dot(x, beta)).ravel().reshape(-1, 1)
    return alpha * m * t ** (alpha - 1) * np.exp(-m * t ** alpha)


X = generating_data(n, p)
T = generate_time_of_event(X, alpha, beta) * 100
R = generate_time_censoring(X, alpha, beta) * 100
L = generate_left_truncation(X, 1, 1.5)
Y = T >= R

loc = R > L


X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])]).loc[loc]
Y = pd.Series(Y, name="target").loc[loc]
L = pd.Series(L, name="left_truncation").loc[loc]
R = pd.Series(R, name="right_censoring").loc[loc]


if __name__ == '__main__':
    import matplotlib.pyplot as plot

    f_xt = density_function(X, np.linspace(0, 1, num=200), alpha, beta)
    plot.pcolormesh(f_xt)
