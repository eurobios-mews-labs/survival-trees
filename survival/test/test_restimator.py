import numpy as np
import pandas as pd
import pytest
from lifelines.utils import concordance_index

from survival import r_estimator


@pytest.fixture(scope="module")
def get_data():
    from lifelines import datasets
    from sklearn.model_selection import train_test_split
    n = 300
    y_cols = ["start_year", "age", "observed"]
    X = pd.DataFrame(dict(x1=np.random.uniform(size=n),
                          x2=np.random.uniform(size=n)), index=range(n))
    y = pd.DataFrame(columns=y_cols, index=X.index)
    y["age"] = X["x1"] * 10 + np.random.uniform(size=n)
    y['start_year'] = 0
    y['observed'] = (y["age"] + np.random.uniform(size=n) + X["x2"]) > 6
    y['observed'] = y['observed'].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(X, y)
    print(y)
    return x_train, x_test, y_train, y_test


def test_r_estimator(get_data):
    rest = r_estimator.REstimator()
    x_train, x_test, y_train, y_test = get_data
    rest._send_to_r_space(x_train, y_train)
    r_df = rest._r_data_frame

    pd_from_r_df = rest._get_from_r_space(["data.X"])["data.X"]
    assert isinstance(pd_from_r_df, pd.DataFrame)


def test_random_forest_src(get_data):
    x_train, x_test, y_train, y_test = get_data
    rf_src = r_estimator.RandomForestSRC(n_estimator=10)
    rf_src.fit(x_train, y_train.drop(columns=["start_year"]))
    pred = rf_src.predict(x_test)
    assert sum(pred.index != x_test.index) == 0
    assert len(rf_src.feature_importances_) == x_train.shape[1]


def test_ltrc_trees(get_data):
    x_train, x_test, y_train, y_test = get_data
    est = r_estimator.LTRCTrees()

    y_save = y_train.__deepcopy__()
    est.fit(x_train, y_train)
    assert sum(y_train.columns == y_save.columns) == y_train.shape[1]


def test_ltrc_trees_n(get_data):
    x_train, x_test, y_train, y_test = get_data

    est = r_estimator.LTRCTrees(
        get_dense_prediction=False,
        interpolate_prediction=True)
    est.fit(x_train, y_train)
    test = est.predict(x_test)
    test[0] = 1
    test = test[np.sort(test.columns)]
    test = test.fillna(method="pad")
    c_index = pd.Series(index=test.columns)
    print(test)
    for date in c_index.index:
        try:
            c_index.loc[date] = concordance_index(
                date * np.ones(len(test)),
                test[date], y_test["observed"])
        except:
            pass
    print(c_index.mean())
    assert c_index.mean() > 0.5


def test_ltrc_trees_predict_curves(get_data):
    x_train, x_test, y_train, y_test = get_data

    est = r_estimator.LTRCTrees()
    est.fit(x_train, y_train)
    curves, indexes = est.predict_curves(x_test)
    print(curves)
    print(curves.drop_duplicates())
    print(indexes)


def test_rf_ltrc(get_data):
    x_train, x_test, y_train, y_test = get_data
    est = r_estimator.RandomForestLTRC(
         n_estimator=3)
    est.fit(x_train, y_train)
    test = est.predict(x_test)

    test[0] = 1
    test = test[np.sort(test.columns)]
    test = test.fillna(method="pad")
    c_index = pd.Series(index=test.columns)

    for date in c_index.index:
        try:
            c_index.loc[date] = concordance_index(
                date * np.ones(len(test)),
                test[date], y_test["observed"])
        except:
            pass
    print(c_index.mean())
    assert c_index.mean() > 0.5


def test_rf_ltrc_fast(get_data):
    x_train, x_test, y_train, y_test = get_data
    est = r_estimator.RandomForestLTRC(n_estimator=30,
                                       max_features=2,
                                       bootstrap=False
                                       )

    est.fit(x_train, y_train)
    test = est.predict(x_test)
    test[0] = 1
    test = test[np.sort(test.columns)]
    test = test.fillna(method="pad")
    c_index = pd.Series(index=test.columns)
    print(test, x_test)
    for date in c_index.index:
        try:
            c_index.loc[date] = concordance_index(
                date * np.ones(len(test)),
                test[date], y_test["observed"])
        except:
            pass
    print(c_index.mean())
    assert c_index.mean() > 0.6
