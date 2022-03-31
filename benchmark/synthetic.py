import numpy as np
import pandas as pd
from scipy import stats

n = 20000
p = 3
beta = np.random.uniform(size=p).reshape(-1, 1)
beta = np.linspace(0, 1, num=p)*10
alpha = 1


# https://onlinelibrary.wiley.com/doi/full/10.1002/sim.9136

def generating_data(n, p):
    pi_k = np.random.uniform(0.2, 0.8, size=p)
    mu_k = stats.norm.ppf(pi_k)
    sigma_ = np.random.normal(size=p * p).reshape(p, p) / 100
    s = np.dot(sigma_.T, sigma_)
    d = s.diagonal()
    sigma = s / (np.sqrt(d).reshape(-1, 1) * np.sqrt(d).reshape(1, -1))
    mg = stats.multivariate_normal(mu_k, sigma, allow_singular=True)
    x_tilde = mg.rvs(size=n) > 0
    return x_tilde


def generate_time_of_event(X, alpha, beta):
    m = np.exp(alpha) * np.exp(np.dot(X, beta))
    m = m.ravel()
    return (-np.log(np.random.uniform(size=len(X)))) ** (1 / alpha) * m


def generate_time_censoring(X, alpha, beta):
    m = np.mean(np.exp(alpha + np.dot(X, beta)).ravel())
    return (-np.log(np.random.uniform(size=len(X)))) ** (1 / alpha) * m


def generate_left_truncation(x, mu, sigma):
    return np.random.lognormal(mu, sigma, size=len(x))


def density_function(X, t, alpha, beta):
    t = t.reshape(1, -1)
    m = np.exp(alpha + np.dot(X, beta)).ravel().reshape(-1, 1)
    return alpha * m * (t ** (alpha - 1)) * np.exp(-m * (t ** alpha))


X = generating_data(10 * n, p)
T = generate_time_of_event(X, alpha, beta)
R = generate_time_censoring(X, alpha, beta)
L = generate_left_truncation(X, np.max((R.mean() - R.std(), T.min())), 1.5)
# L = 0 * np.ones(L.shape)
Y = T <= R
Y &= T > L

print(np.mean(Y))
print(np.mean(T))

R = np.where(Y, T, R)
loc = (R - L) > 1e-3

X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])]).loc[loc].iloc[:n]
Y = pd.Series(Y, name="target").loc[loc].iloc[:n]
L = pd.Series(L, name="left_truncation").loc[loc].iloc[:n]
R = pd.Series(R, name="right_censoring").loc[loc].iloc[:n]

if __name__ == '__main__':
    import matplotlib.pyplot as plot
    import seaborn as sns
    from survival_trees import RandomForestLTRCFitter, LTRCTreesFitter
    from survival_trees.metric import concordance_index
    from lifelines.fitters import coxph_fitter, log_logistic_aft_fitter
    from sklearn.model_selection import train_test_split

    t = np.linspace(min(R), max(R), num=200)
    f_xt = density_function(X, t, alpha, beta)
    plot.pcolormesh(f_xt)
    models = {
        "ltrc-forest": RandomForestLTRCFitter(
            n_estimators=20,
            min_samples_leaf=3,
            max_samples=0.8),
        "ltrc_trees": LTRCTreesFitter(),
        "cox-semi-parametric": coxph_fitter.SemiParametricPHFitter(),
        "aft-log-logistic": log_logistic_aft_fitter.LogLogisticAFTFitter(),
    }
    data = pd.concat((X, Y, L,
                      R), axis=1).astype(float)

    sns.heatmap(
        pd.concat((data.iloc[:, -12:],
                   pd.Series(T, name='time')), axis=1).corr(method="spearman"), vmin=-1,
        cmap='RdBu')
    plot.tight_layout()
    y = data[["left_truncation", "right_censoring", "target"]]
    X = data.drop(columns=y.columns.tolist())
    X = X.select_dtypes(include=np.number)
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, train_size=0.6)
    for i, key in enumerate(models.keys()):
        models[key].fit(
            pd.concat((x_train, y_train), axis=1).dropna(),
            entry_col=y_train.columns[0],
            duration_col=y_train.columns[1],
            event_col=y_train.columns[2]
        )
        test = 1 - models[key].predict_cumulative_hazard(
            x_test).astype(float).T
        test = test.dropna()
        c_index = concordance_index(
            test, death=y_test.loc[test.index].iloc[:, 2],
            censoring_time=y_test.loc[test.index].iloc[:, 1])
        print(models[key])
        print(c_index.mean())
