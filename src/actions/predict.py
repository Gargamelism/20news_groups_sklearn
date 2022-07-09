from sklearn.datasets import fetch_20newsgroups
import numpy
from actions.train import get_classifier


def predict(parsed_args):
    test_data = fetch_20newsgroups(subset="test")
    classifier_wrapper = get_classifier(parsed_args)

    transformed_test_data = classifier_wrapper.data_transformer(test_data.data)
    predicted = classifier_wrapper.classifier.predict(transformed_test_data)
    print(numpy.mean(predicted == test_data.target))
