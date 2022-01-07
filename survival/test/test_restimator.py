import numpy as np
import pandas as pd
import pytest
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

from survival import r_estimator


@pytest.fixture(scope="module")
def get_data():
    from lifelines import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder
    ohe = OneHotEncoder(sparse=False)
    dataset = datasets.load_dd()
    y_cols = ["start_year", "age", "observed"]
    X = pd.DataFrame(dict(x1=np.random.uniform(size=1000),
                          x2=np.random.uniform(size=1000)), index=range(1000))
    y = pd.DataFrame(columns=y_cols, index=X.index)
    y["age"] = X["x1"] * 10 + np.random.uniform(size=1000)
    y['start_year'] = 0
    y['observed'] = (y["age"] + np.random.uniform(size=1000)) > 6
    y['observed'] = y['observed'].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(X, y)
    return x_train, x_test, y_train, y_test


def test_r_estimator(get_data):
    rest = r_estimator.REstimator()
    x_train, x_test, y_train, y_test = get_data
    rest.send_to_r_space(x_train, y_train)
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


