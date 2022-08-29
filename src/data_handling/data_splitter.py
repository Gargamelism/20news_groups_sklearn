import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

from src.ml_modeling.types.model_data_wrapper import ModelSplitWrapper


def split_data(train_data: pd.Series, target: pd.Series) -> Tuple[
    ModelSplitWrapper, ModelSplitWrapper, ModelSplitWrapper]:
    X_train, X_test, y_train, y_test = train_test_split(train_data, target, test_size=0.3, random_state=42)
    X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

    return (
        ModelSplitWrapper(X_train, y_train), ModelSplitWrapper(X_validate, y_validate),
        ModelSplitWrapper(X_test, y_test)
        )
