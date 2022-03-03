#
import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import datasets
from lifelines.fitters import coxph_fitter, log_logistic_aft_fitter
from lifelines.plotting import plot_lifetimes
from sklearn.model_selection import train_test_split

from survival_trees import LTRCTrees, RandomForestLTRCFitter, RandomForestLTRC, LTRCTreesFitter
from survival_trees import plotting
from survival_trees.metric import concordance_index, time_dependent_roc


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
    # ==========================================================================
    data = datasets.load_gbsg2()
    data["entry_date"] = 0
    y = data[["entry_date", "time", "cens"]]
    X = data.drop(columns=y.columns.tolist())
    X = X.select_dtypes(include=np.number)
    datasets_dict["gbsg2"] = X, y
    # ==========================================================================
    return datasets_dict


def benchmark():
    all_datasets = load_datasets()
    models = {
        "ltrc-forest": RandomForestLTRCFitter(
            n_estimators=20,
            min_samples_leaf=3,
            max_samples=0.8),
        "ltrc_trees": LTRCTreesFitter(),
        "cox-semi-parametric": coxph_fitter.SemiParametricPHFitter(),
        "aft-log-logistic": log_logistic_aft_fitter.LogLogisticAFTFitter(),
    }
    results = pd.DataFrame(index=all_datasets.keys(), columns=models.keys())

    for k, (X, y) in all_datasets.items():
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
            results.loc[k, key] = np.nanmean(c_index)
    print(results)
    f, ax = plot.subplots(figsize=(9, 6))
    sns.heatmap(results.astype(float), annot=True, linewidths=2, ax=ax,
                vmin=0.5, vmax=0.9)
    plot.savefig("./public/benchmark.png")


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
