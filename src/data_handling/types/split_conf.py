from dataclasses import dataclass
from typing import Union

from src.data_handling.types.split_by_enum import SplitByEnum

@dataclass
class SplitConf:
    split_by: SplitByEnum = SplitByEnum.SHUFFLE
    random_state: Union[int, bool] = 42
    test_size: int = 0.3