import pandas as pd
from dataclasses import dataclass


@dataclass
class ModelSplitWrapper:
    data: pd.Series
    target: pd.Series


@dataclass
class ModelDataWrapper:
    data: pd.DataFrame
    target: pd.Series
