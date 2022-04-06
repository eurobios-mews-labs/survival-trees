import numpy as np
import pandas as pd
import pytest
from lifelines.utils import concordance_index

from survival_trees import LTRCTrees, RandomForestSRC, RandomForestLTRC
from survival_trees import _base


def test_auc(get_data):
    pass