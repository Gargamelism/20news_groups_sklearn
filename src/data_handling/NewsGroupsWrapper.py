import pandas as pd
from dataclasses import dataclass
from typing import List


@dataclass
class NewsGroupsWrapper:
    data: pd.DataFrame
    group_names: List[str]
