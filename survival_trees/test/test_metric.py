import numpy as np
import pandas as pd
import pytest
from lifelines.utils import concordance_index

from survival_trees import LTRCTrees, RandomForestSRC, RandomForestLTRC
from survival_trees import _base

from survival_trees import metric


def test_auc():
    t1 = np.linspace(0, 1, num=100)
    t2 = np.linspace(1, 0, num=100)
    t3 = np.linspace(2, 0, num=100)

    data = pd.DataFrame(np.array([t1, t2, t3]))
    target = [0, 1, 1]
    metric.time_dependent_auc(data, target, [100, 75, 100])