import pandas as pd
from enum import IntEnum, auto
from dataclasses import dataclass
from typing import Any

from src.ml_modeling.types.model_classifier_enum import ModelClassifierEnum


class FeatureImportanceEnum(IntEnum):
    FEATURES = auto()
    IMPORTANCE = auto()


@dataclass
class ClassifierWrapper:
    type: ModelClassifierEnum
    classifier: Any
    feature_importance: pd.DataFrame = None
