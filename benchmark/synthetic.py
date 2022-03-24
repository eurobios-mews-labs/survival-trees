import numpy as np
from scipy import stats


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


if __name__ == '__main__':
    n = 10000
    p = 3
    X = generating_data(n, p)
    beta = np.random.uniform(size=p).reshape(-1, 1)
    alpha = np.random.uniform(size=p).reshape(-1, 1)
    T = generate_time_of_event(X, alpha, beta)
