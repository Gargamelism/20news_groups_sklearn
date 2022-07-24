from sklearn.datasets import fetch_20newsgroups
import numpy

from data_classes import ParsedArgs
from actions.data_preprocess import preprocess_data
from actions.data_format import get_data_formatter_wrapper
from actions.train import get_trained_classifier


def test(parsed_args: ParsedArgs):
    dataset = fetch_20newsgroups()

    preprocessed_data = preprocess_data(parsed_args, dataset.data)
    data_formatter = get_data_formatter_wrapper(parsed_args, preprocessed_data)
    formatted_data = data_formatter.format(preprocessed_data)

    classifier = get_trained_classifier(parsed_args, formatted_data, dataset.target)

    test_dataset = fetch_20newsgroups(subset="test")
    preprocessed_test_data = preprocess_data(parsed_args, test_dataset.data)
    formatted_test_data = data_formatter.format(preprocessed_test_data)

    predict_proba = classifier.predict_proba(formatted_test_data)
    print(predict_proba)
