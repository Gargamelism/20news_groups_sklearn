from dataclasses import dataclass
from typing import List


@dataclass
class ParsedArgs:
    show_target_distribution: bool
    sample_size: int
    data_processors: List[str]
    data_vectorizers: List[str]
    classifiers: List[str]
