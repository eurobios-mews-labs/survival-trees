import numpy as np
import pandas as pd

from survival import r_estimator as re


def test_ltrc_2():
    from lifelines import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder
    from lifelines.utils import concordance_index
    ohe = OneHotEncoder(sparse=False)
    dataset = datasets.load_dd()
    y_cols = ["start_year", "age", "observed"]
    X = pd.DataFrame(dict(x1=np.random.uniform(size=1000),
                          x2=np.random.uniform(size=1000)), index = range(1000))
    y = pd.DataFrame(columns=y_cols, index=X.index)
    y["age"] = X["x1"] * 10 + np.random.uniform(size=1000)
    y['start_year'] = 0
    y['observed'] = (y["age"] + np.random.uniform(size=1000)) > 6
    y['observed'] = y['observed'].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(X, y)

    model2 = re.RandomForestLTRC(n_estimator=30, n_features=x_train.shape[1])
    model1 = re.LTRCTrees()

    for i, model in enumerate([model1, model2]):
        model.fit(x_train, y_train)

        test = model.predict(x_test).astype(float)
        # test.T.plot(legend=False, cmap="jet")

        c_index = pd.Series(index=test.columns)
        for date in c_index.index:
            try:
                c_index.loc[date] = concordance_index(
                    date * np.ones(len(test)),
                     test[date], y_test["observed"])
            except:
                pass

        c_index.plot(label=str(i), legend=True)

        self = model


def test_ltrc():
    from lifelines import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder
    from lifelines.utils import concordance_index
    ohe = OneHotEncoder(sparse=False)
    dataset = datasets.load_dd()
    y_cols = ["start_year", "duration", "observed"]
    X, y = dataset.drop(y_cols, axis=1), dataset[y_cols]
    ohe.fit(X[["un_continent_name", "regime"]])
    columns = [l for l in ohe.categories_]
    for i, col in enumerate(columns):

        cols = list(col)
        for col in cols:
            X[col] = 1

    X[np.concatenate(columns)] = ohe.transform(
        X[["un_continent_name", "regime"]])

    X = X._get_numeric_data()
    X = X.dropna()
    y = y.loc[X.index]
    y["age"] = y["start_year"] + y["duration"]
    y = y[["start_year", "age", "observed"]]
    x_train, x_test, y_train, y_test = train_test_split(X, y)

    model2 = re.RandomForestLTRC(n_estimator=100, n_features=x_train.shape[1],
                                 bootstrap=False)
    model1 = re.LTRCTrees()

    for i, model in enumerate([model1, model2]):
        model.fit(x_train, y_train)

        test = model.predict(x_test).astype(float)
        # test.T.plot(legend=False, cmap="jet")

        c_index = pd.Series(index=test.columns)
        for date in c_index.index:
            try:
                c_index.loc[date] = concordance_index(
                    date * np.ones(len(test)),
                     test[date], y_test["observed"])
            except:
                pass

        c_index.plot(label=str(i), legend=True)

    self = model2
