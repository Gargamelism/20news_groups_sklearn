from dataclasses import dataclass, field
from typing import List, Set

from src.data_handling.types.data_cleaning_enum import DataCleaningEnum
from src.data_handling.types.split_by_enum import SplitByEnum


@dataclass
class ParsedArgs:
    show_target_distribution: bool
    sample_size: int
    data_vectorizers: List[str]
    classifiers: List[str]
    print_feature_importance: bool
    split_validation: bool

    # special parsings
    data_processors: Set[DataCleaningEnum]
    _data_processors: Set[DataCleaningEnum] = field(init = False)

    separate_train_test: SplitByEnum
    _separate_train_test: SplitByEnum = field(init = False)

    @property
    def data_processors(self) -> Set[DataCleaningEnum]:
        return self._data_processors

    @data_processors.setter
    def data_processors(self, data_processors: List[str]):
        self._data_processors = {DataCleaningEnum[data_processor.upper()] for data_processor in data_processors}

    @property
    def separate_train_test(self) -> SplitByEnum:
        return self._separate_train_test

    @separate_train_test.setter
    def separate_train_test(self, separate_train_test: str):
        self._separate_train_test = SplitByEnum[
            separate_train_test.upper()] if type(separate_train_test) == str else separate_train_test
