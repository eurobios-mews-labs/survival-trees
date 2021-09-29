import numpy as np

from survival import r_estimator as re
import pandas as pd

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

    model2 = re.RandomForestLTRC(n_estimator=1, n_features=x_train.shape[1])
    model1 = re.LTRCTrees()

    for model in [model1, ]:
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

        c_index.plot()
    self = model2
