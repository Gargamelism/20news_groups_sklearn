import pandas as pd
from dataclasses import dataclass
from typing import List


@dataclass
class NewsGroupsWrapper:
    orig_data: pd.DataFrame
    group_names: List[str]
    data: pd.DataFrame
