"""
https://medium.com/analytics-vidhya/concordance-index-72298c11eac7
https://anaqol.org/ea4275/js2012/3-Combescure.pdf?PHPSESSID=9fe65030a58af33d1370d7ccff343fa4
"""
import numpy as np
import pandas as pd
from lifelines import utils
from sklearn import metrics as sk_metrics


def concordance_index(temporal_score: pd.DataFrame, death: iter,
                      censoring_time=iter):
    c_index = pd.Series(index=temporal_score.columns)
    for date in c_index.index:
        try:
            c_index.loc[date] = utils.concordance_index(
                censoring_time,
                temporal_score[date], death & (date >= censoring_time))
        except ZeroDivisionError:
            pass

    return c_index


def mean_concordance_index(temporal_score: pd.DataFrame, death: iter,
                           censoring_time=iter):
    return concordance_index(temporal_score, death,
                             censoring_time).mean()


def time_dependent_roc(
        temporal_score: pd.DataFrame,
        death: iter,
        censoring_time=iter,
        method="harrell"
):
    """
    method
        - roc-cd : Cumulative sensitivity and dynamic specificity (C/D)
        - roc-id : Incident sensitivity and dynamic specificity (I/D)
    """
    tdr = pd.Series(index=temporal_score.columns)

    if method == "harrell":
        def outcome(_):
            return death

        def marker(t_):
            return temporal_score[t_]

    elif method == "roc-cd":
        def outcome(t_):
            return (censoring_time <= t_) & death

        def marker(t_):
            return temporal_score[t_]

    elif method == "roc-id":
        def outcome(t_):
            out = np.where(censoring_time < t_, np.nan, censoring_time)
            return (t_ == out) & death

        def marker(t_):
            return temporal_score[t_]
    else:
        raise ValueError(f"method : {method} is not known")

    for t in tdr.index:
        try:
            y_true = np.array(outcome(t))
            y_score = np.array(marker(t))

            nan_ = np.isnan(y_true)
            nan_ |= np.isnan(y_score)

            tdr.loc[t] = sk_metrics.roc_auc_score(
                y_true=y_true[~nan_],
                y_score=y_score[~nan_]
            )
        except ValueError:
            pass
    return tdr
