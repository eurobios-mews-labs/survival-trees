import matplotlib.pyplot as plot
import numpy as np
import pandas as pd


def tagged_curves(temporal_curves: pd.DataFrame, label: pd.DataFrame,
                  time_event: pd.Series = None,
                  event_observed_color="#A60628",
                  event_censored_color="#348ABD",
                  add_marker=False
                  ):
    temp_curves = temporal_curves.astype("float16")
    temp_curves.index = range(len(temp_curves))
    dates = temp_curves.columns.to_list()
    pos_index = temp_curves.index[label.astype(bool)]
    neg_index = temp_curves.index[~label.astype(bool)]
    # check consistency
    if time_event is not None:
        col_id = np.searchsorted(temp_curves.columns, time_event, side="right") - 1
        col = temp_curves.columns[col_id]
        if add_marker:
            prob = np.diag(temp_curves[col])
            plot.scatter(col[label.astype(bool)], prob[label.astype(bool)],
                         marker="*",
                         c=event_observed_color)
        outdated = pd.DataFrame(np.nan, index=temp_curves.index,
                                columns=temp_curves.columns)
        for i, ind in enumerate(temp_curves.index):
            data = temp_curves.loc[ind]
            outdated.loc[ind, dates[col_id[i]:]] = data[dates[col_id[i]:]]
            temp_curves.loc[ind, dates[col_id[i] + 1:]] = np.nan
        plot_args = dict(alpha=0.4, ls="--", lw=1)
        plot.plot(outdated.loc[neg_index].T, c=event_censored_color,
                  **plot_args)
        plot.plot(outdated.loc[pos_index].T, c=event_observed_color,
                  **plot_args)
    plot_args = dict(lw=1, alpha=0.9)
    plot.plot(temp_curves.loc[pos_index].T, c=event_observed_color, **plot_args)
    plot.plot(temp_curves.loc[neg_index].T, c=event_censored_color, **plot_args)
