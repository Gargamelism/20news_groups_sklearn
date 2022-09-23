import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from typing import Any, List
from pprint import pprint

from src.ml_modeling.types.model_data_wrapper import ModelDataWrapper
from src.ml_modeling.types.classifier_wrapper import ClassifierWrapper

def evaluate_baseline(target: pd.DataFrame, target_names: List[str]):
    prob_baseline = target.value_counts(normalize=True)
    target_baseline = np.random.choice(prob_baseline.index, size=target.size, p=prob_baseline.values)
    print(classification_report(target, target_baseline, target_names=target_names))


def evaluate(classifier: ClassifierWrapper, test_data_wrapper: ModelDataWrapper, target_names: List[str]):
    print(classifier.type)
    classifier_predictions = classifier.classifier.predict(test_data_wrapper.data)
    evaluation_result = classification_report(test_data_wrapper.target, classifier_predictions, target_names=target_names, zero_division=0, output_dict=True)
    pprint(evaluation_result, sort_dicts=True)
    return evaluation_result
