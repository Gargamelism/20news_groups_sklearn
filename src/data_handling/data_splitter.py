import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

from src.ml_modeling.types.ModelDataWrapper import ModelDataWrapper


def split_data(train_data: pd.Series, target: pd.Series) -> Tuple[
    ModelDataWrapper, ModelDataWrapper, ModelDataWrapper]:
    X_train, X_test, y_train, y_test = train_test_split(train_data, target, test_size=0.3, random_state=42)
    X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

    return (
        ModelDataWrapper(X_train, y_train), ModelDataWrapper(X_validate, y_validate), ModelDataWrapper(X_test, y_test)
        )
