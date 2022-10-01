import pandas as pd
from dataclasses import dataclass


@dataclass
class ModelSplitWrapper:
    data: pd.Series
    target: pd.Series
    split_vars: pd.Series = None


@dataclass
class ModelDataWrapper:
    data: pd.DataFrame
    target: pd.Series
