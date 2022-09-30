import pandas as pd
from dataclasses import dataclass
from typing import Union

from src.data_handling.types.split_by_enum import SplitByEnum


@dataclass
class SplitConf:
    split_by: SplitByEnum = SplitByEnum.DEFAULT
    split_by_vals: pd.Series = pd.Series([])
    random_state: Union[int, bool] = 42
    test_size: int = 0.3
