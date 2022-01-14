from lifelines import datasets
from sklearn.model_selection import train_test_split

from survival_trees import LTRCTrees, RandomForestLTRC
from survival_trees.metric import concordance_index, time_dependent_roc


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
        # test.T.plot(legend=False, cmap="jet")

        c_index = concordance_index(
            test, death=y_test["death"],
            censoring_time=y_test["time"])
        c_index.plot()


def test_metrics():
    data = datasets.load_larynx()
    data["entry_date"] = 0
    y = data[["entry_date", "time", "death"]]
    X = data.drop(columns=y.columns.tolist())

    model = RandomForestLTRC(max_features=2, n_estimators=30,
                               min_samples_leaf=4)

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8)
    model = RandomForestLTRC(max_features=2, n_estimators=30,
                             min_samples_leaf=4)
    model.fit(x_train, y_train)
    for method in ["harrell", "roc", "incidence-roc"]:
        test = model.predict(x_test).astype(float)
        tdr = time_dependent_roc(test, death=y_test["death"],
                                 censoring_time=y_test["time"],
                                 method=method)
        tdr.plot(label=method)
