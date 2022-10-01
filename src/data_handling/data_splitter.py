import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List
from enum import IntEnum, auto

from src.ml_modeling.types.model_data_wrapper import ModelSplitWrapper
from src.data_handling.types.split_conf import SplitConf
from src.data_handling.types.split_by_enum import SplitByEnum
from src.data_handling.types.news_groups_data_enum import NewsGroupsDataEnum


def split_data(train_data: pd.Series, target: pd.Series, conf: SplitConf = SplitConf()) -> Tuple[
    ModelSplitWrapper, ModelSplitWrapper]:
    new_data_splitter = DataSplitter(train_data, target)

    return new_data_splitter.split_data(conf)


class SplitClassesEnum(IntEnum):
    TRAIN = auto()
    TEST = auto()


class DataSplitter:
    def __init__(self, train_data: pd.Series, target: pd.Series):
        self.data = train_data
        self.target = target

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, new_data: pd.Series):
        self._data = new_data

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, new_target: pd.Series):
        self._target = new_target

    def split_data(self, conf: SplitConf = SplitConf()) -> Tuple[ModelSplitWrapper, ModelSplitWrapper]:
        train_test_split_args = {
            'test_size': conf.test_size,
            'random_state': conf.random_state
            }

        if conf.split_by == SplitByEnum.EMAILS:
            X_train, X_test, y_train, y_test, train_split_vals, test_split_vals = self.__train_test_split_by_vals(
                conf.split_by_vals, train_test_split_args
                )
            train_split_wrapper = ModelSplitWrapper(X_train, y_train, pd.Series(train_split_vals))
            test_split_wrapper = ModelSplitWrapper(X_test, y_test, pd.Series(test_split_vals))
        else:
            X_train, X_test, y_train, y_test = train_test_split(self.data, self.target, **train_test_split_args)
            train_split_wrapper = ModelSplitWrapper(X_train, y_train)
            test_split_wrapper = ModelSplitWrapper(X_test, y_test)

        return train_split_wrapper, test_split_wrapper

    def __train_test_split_by_vals(self, vals: pd.Series, train_test_split_args: Dict) -> Tuple[
        pd.Series, pd.Series, pd.Series, pd.Series, List[str], List[str]]:
        SPLIT_COL_KEY = 'SPLIT_COL_KEY'

        train_split_vals, test_split_vals = train_test_split(vals, **train_test_split_args)

        split_on_vals = self.data.map(
            lambda text: SplitClassesEnum.TEST if any(
                test_val for test_val in test_split_vals if test_val in text
                ) else SplitClassesEnum.TRAIN
            )

        train_test_groups = pd.DataFrame(
            {NewsGroupsDataEnum.DATA: self.data, NewsGroupsDataEnum.TARGET: self.target, SPLIT_COL_KEY: split_on_vals}
            ).groupby(SPLIT_COL_KEY)

        train_df = train_test_groups.get_group(SplitClassesEnum.TRAIN)
        test_df = train_test_groups.get_group(SplitClassesEnum.TEST)

        return (
            train_df[NewsGroupsDataEnum.DATA], test_df[NewsGroupsDataEnum.DATA],
            train_df[NewsGroupsDataEnum.TARGET], test_df[NewsGroupsDataEnum.TARGET],
            train_split_vals, test_split_vals
            )
