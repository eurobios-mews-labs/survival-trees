import matplotlib
#
import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
from lifelines import datasets
from lifelines.plotting import plot_lifetimes
from sklearn.model_selection import train_test_split

from survival_trees import LTRCTrees, RandomForestLTRC
from survival_trees import plotting
from survival_trees.metric import concordance_index, time_dependent_roc

from lifelines.fitters import coxph_fitter

matplotlib.use('agg')


def load_datasets():
    datasets_dict = {}
    # ==========================================================================
    data = datasets.load_larynx()
    data["entry_date"] = 0
    y = data[["entry_date", "time", "death"]]
    X = data.drop(columns=y.columns.tolist())
    datasets_dict["larynx"] = X, y
    # ==========================================================================
    data = datasets.load_dd()
    data["entry_date"] = 0
    y = data[["entry_date", "duration", "observed"]]
    X = data.drop(columns=y.columns.tolist())
    X = X.select_dtypes(include=np.number)
    datasets_dict["dd"] = X, y
    # ==========================================================================
    data = datasets.load_lung()
    data["entry_date"] = 0
    y = data[["entry_date", "time", "status"]]
    X = data.drop(columns=y.columns.tolist())
    X = X.select_dtypes(include=np.number)
    datasets_dict["lung"] = X, y
    # ==========================================================================
    data = datasets.load_rossi()
    data["entry_date"] = 0
    y = data[["entry_date", "week", "arrest"]]
    X = data.drop(columns=y.columns.tolist())
    X = X.select_dtypes(include=np.number)
    datasets_dict["rossi"] = X, y
    return datasets_dict


def benchmark():
    all_datasets = load_datasets()
    models = (RandomForestLTRC(max_features=2, n_estimators=50,
                               min_samples_leaf=4, max_samples=0.8),
              LTRCTrees(min_samples_leaf=4),
              coxph_fitter.SemiParametricPHFitter())

    results = pd.DataFrame(index=all_datasets.keys(), columns=models)
    for k, (X, y) in all_datasets.items():
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, train_size=0.8)
        for i, model in enumerate(models):
            model.fit(x_train, y_train)
            test = model.predict(x_test).astype(float)
            c_index = concordance_index(
                test, death=y_test.iloc[:, 2],
                censoring_time=y_test.iloc[:, 1])
            results.loc[k, model] = c_index.mean()


def test_larynx():
    data = datasets.load_larynx()
    data["entry_date"] = 0
    y = data[["entry_date", "time", "death"]]
    X = data.drop(columns=y.columns.tolist())

    models = (RandomForestLTRC(max_features=2, n_estimators=30,
                               min_samples_leaf=4),
              LTRCTrees(min_samples_leaf=4))
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8)

    for i, model in enumerate(models):
        model.fit(x_train, y_train)
        test = model.predict(x_test).astype(float)
        c_index = concordance_index(
            test, death=y_test["death"],
            censoring_time=y_test["time"])
        c_index.plot()


def test_metrics():
    from importlib import reload
    reload(plotting)
    data = datasets.load_larynx()
    data["entry_date"] = 0
    y = data[["entry_date", "time", "death"]]
    X = data.drop(columns=y.columns.tolist())

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8)
    model = RandomForestLTRC(max_features=2, n_estimators=30,
                             min_samples_leaf=4)
    model.fit(x_train, y_train)
    plot.figure()
    for method in ["harrell", "roc-cd", "roc-id"]:
        test = model.predict(x_test).astype(float)
        tdr = time_dependent_roc(1 - test, death=y_test["death"],
                                 censoring_time=y_test["time"],
                                 method=method)
        tdr.dropna().plot(marker=".", label=method)
    plot.legend()
    plot.savefig("benchmark/metric.png")

    plot.figure()
    plot_lifetimes(y["time"], entry=y["entry_date"], event_observed=y["death"])
    plot.savefig("benchmark/lifelines.png")

    plot.figure()

    plotting.tagged_curves(temporal_curves=test, label=y_test["death"],
                           time_event=y_test["time"],
                           add_marker=False)
    plot.savefig("benchmark/curves.png", dpi=200)
