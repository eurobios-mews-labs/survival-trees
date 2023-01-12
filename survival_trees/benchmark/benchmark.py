import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import datasets
from lifelines.fitters import coxph_fitter, log_logistic_aft_fitter
from lifelines.plotting import plot_lifetimes
from sklearn.model_selection import train_test_split

from survival_trees.benchmark import synthetic
from survival_trees import LTRCTrees, RandomForestLTRCFitter, RandomForestLTRC, LTRCTreesFitter
from survival_trees import plotting
from survival_trees.metric import concordance_index, time_dependent_auc

import rpy2
from rpy2 import robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
plot.rc('font', family='ubuntu')


def load_datasets():
    datasets_dict = {}
    # ==========================================================================
    data = datasets.load_larynx()
    data["entry_date"] = data["age"]
    data["time"] += data["entry_date"]
    y = data[["entry_date", "time", "death"]]
    X = data.drop(columns=y.columns.tolist())
    datasets_dict["Larynx Cancer"] = X, y
    # ==========================================================================
    data = datasets.load_lung()
    data["entry_date"] = data["age"] * 365.25
    data["time"] += data["entry_date"]
    y = data[["entry_date", "time", "status"]]
    X = data.drop(columns=y.columns.tolist())
    X = X.select_dtypes(include=np.number)
    datasets_dict["Lung Cancer"] = X, y
    # ==========================================================================
    data = datasets.load_gbsg2().dropna()
    data["death"] = 1 - data["cens"]
    data = data.drop(columns='cens', axis=1)
    data["entry_date"] = data["age"]
    data["time"] /= 365.25
    data["time"] += data["entry_date"]

    data["horTh"] = data["horTh"] == "yes"
    data["menostat"] = data["menostat"] == "Post"
    data["tgrade"] = data["tgrade"] == "III"
    y = data[["entry_date", "time", "death"]].copy()
    X = data.drop(columns=y.columns.tolist())
    X = X.astype(float).select_dtypes(include=np.number)
    datasets_dict["Breast Cancer"] = X, y
    # =========================================================================
    robjects.r("library(survival)")
    with localconverter(robjects.default_converter + pandas2ri.converter):
        data = robjects.conversion.rpy2py(robjects.r("survival::flchain"))
    data["sex"] = (data["sex"] == "F").astype(int)
    y = pd.DataFrame(index=data.index)
    data.loc[data["futime"] <= 0, "futime"] = 0.5
    y["time"] = data["futime"]/365.25 + data["age"]
    y["death"] = data["death"]
    y["entry_point"] = data["age"]
    y = y[["entry_point", "time", "death"]]
    y['death'] = y["death"].astype(int)
    X = data[["sex", "kappa", "lambda", "creatinine", "mgus"]]
    datasets_dict["FLC chain"] = X, y
    # ==========================================================================
    data = datasets.load_dd()
    data["entry_date"] = 0
    y = data[["entry_date", "duration", "observed"]]
    X = data.drop(columns=y.columns.tolist())
    X = X.select_dtypes(include=np.number)
    datasets_dict["Dictatorship \& Democracy"] = X, y
    # ==========================================================================
    data = datasets.load_rossi()
    data["entry_date"] = 0
    y = data[["entry_date", "week", "arrest"]]
    X = data.drop(columns=y.columns.tolist())
    X = X.select_dtypes(include=np.number)
    datasets_dict["Convicts"] = X, y
    # ==========================================================================
    data = pd.concat((synthetic.X.astype(int), synthetic.Y, synthetic.L,
                      synthetic.R), axis=1)
    y = data[["left_truncation", "right_censoring", "target"]]
    X = data.drop(columns=y.columns.tolist())
    X = X.select_dtypes(include=np.number)
    datasets_dict["Synthetic data"] = X, y
    return datasets_dict


def benchmark(n_exp=2):
    all_datasets = load_datasets()
    models = {
        "ltrc-forest": RandomForestLTRCFitter(
            n_estimators=30,
            min_impurity_decrease=0.0000001,
            min_samples_leaf=3,
            max_samples=0.89),
        "ltrc-trees": LTRCTreesFitter(min_samples_leaf=3, min_impurity_decrease=0.00000001),
        "cox-semi-parametric": coxph_fitter.SemiParametricPHFitter(penalizer=0.1),
        "aft-log-logistic": log_logistic_aft_fitter.LogLogisticAFTFitter(penalizer=0.1),
    }
    results = {}
    for j in range(n_exp):
        results[j] = pd.DataFrame(index=all_datasets.keys(), columns=models.keys())
        for k, (X, y) in all_datasets.items():
            x_train, x_test, y_train, y_test = train_test_split(
                X, y, train_size=0.7)
            for i, key in enumerate(models.keys()):
                pass
                try:
                    models[key].fit(
                        pd.concat((x_train, y_train), axis=1).dropna(),
                        entry_col=y_train.columns[0],
                        duration_col=y_train.columns[1],
                        event_col=y_train.columns[2]
                    )
                    test = - np.log(models[key].predict_cumulative_hazard(
                        x_test).astype(float)).T
                    test = test.dropna()
                    c_index = time_dependent_auc(
                        - test, event_observed=y_test.loc[test.index].iloc[:, 2],
                        censoring_time=y_test.loc[test.index].iloc[:, 1])
                    results[j].loc[k, key] = np.nanmean(c_index)
                except Exception:
                    pass
    all_res = pd.DataFrame()
    for res in results.keys():
        results[res]["num_expe"] = res
        all_res = pd.concat((results[res].astype(float), all_res), axis=0)
    all_res.index.name = "dataset"
    all_res.to_csv("benchmark/benchmark_data.csv")
    mean_ = all_res.reset_index().groupby("dataset").mean().drop(columns=["num_expe"])
    std_ = all_res.reset_index().groupby("dataset").std().drop(columns=["num_expe"])
    return mean_, std_


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
                             min_samples_leaf=1,
                             cp=0.000000001)
    model.fit(x_train, y_train)
    plot.figure()
    for method in ["harrell", "roc-cd", "roc-id"]:
        test = model.predict(x_test).astype(float)
        tdr = time_dependent_auc(1 - test, event_observed=y_test["death"],
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
    # plot.savefig("benchmark/curves.png", dpi=200)


def properties():
    data = load_datasets()
    for k, (X, y) in data.items():
        print(X.shape)




if __name__ == '__main__':
    datasets_dict = load_datasets()
    data_names = list(datasets_dict.keys())
    # mean_, _ = benchmark(n_exp=20)
    # mean_.to_csv("benchmark/benchmark.csv")

    all_res = pd.read_csv("survival_trees/benchmark/benchmark_data.csv", index_col="dataset")
    mean_ = all_res.reset_index().groupby("dataset").mean().drop(columns=["num_expe"])
    std_ = all_res.reset_index().groupby("dataset").std().drop(columns=["num_expe"])

    mean_ = mean_.loc[data_names]
    mean_.index = [e.replace("\&", "&") for e in mean_.index]
    f, ax = plot.subplots(figsize=(4.6, 2.6), dpi=300)
    sns.heatmap(mean_.astype(float), annot=True, linewidths=2, ax=ax,
                vmin=0.5,
                # vmax=0.9,
                cmap="rocket_r"
                )
    plot.xticks(rotation=0)
    import textwrap

    f = lambda x: textwrap.fill(x.get_text(), 12)
    ax.set_yticklabels(map(f, ax.get_yticklabels()))

    f = lambda x: textwrap.fill(x.get_text(), 15)
    ax.set_xticklabels(map(f, ax.get_xticklabels()))

    plot.ylabel("")
    plot.tight_layout()
    plot.savefig("./public/benchmark.png")


    for k, data_ in datasets_dict.items():
        X, y = data_
        y.columns = ["entry_date", "time", "death"]
        pd.concat((X, y), axis=1).iloc[:600].to_csv(f"survival_trees/benchmark/data/{k}.txt", index=False)