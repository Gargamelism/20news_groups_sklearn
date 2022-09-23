import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

from src.ml_modeling.types.model_data_wrapper import ModelSplitWrapper
from src.data_handling.types.split_conf import SplitConf
from src.data_handling.types.split_by_enum import SplitByEnum


def split_data(train_data: pd.Series, target: pd.Series, conf: SplitConf = SplitConf()) -> Tuple[
    ModelSplitWrapper, ModelSplitWrapper]:
    new_data_splitter = DataSplitter(train_data, target)

    return new_data_splitter.split_data(conf)


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
        train_test_args = {
            'test_size': conf.test_size
            }
        if (conf.split_by == SplitByEnum.SHUFFLE):
            train_test_args['random_state'] = conf.random_state

        X_train, X_test, y_train, y_test = train_test_split(self.data, self.target, **train_test_args)

        return (ModelSplitWrapper(X_train, y_train), ModelSplitWrapper(X_test, y_test))
