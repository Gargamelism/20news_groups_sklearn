from sklearn.datasets import fetch_20newsgroups
import numpy
from actions.train import get_classifier


def predict(parsed_args):
    test_data = fetch_20newsgroups(subset="test")
    classifier = get_classifier(parsed_args)
    predicted = classifier.predict(test_data.data)
    print(numpy.mean(predicted == test_data.target))
