from sklearn.datasets import fetch_20newsgroups
import numpy
from actions.train import get_classifier
from helpers.ManipulateData import ManipulateData


def predict(parsed_args):
    test_data = fetch_20newsgroups(subset="test")
    test_data = ManipulateData.text_to_vectors(test_data.data)
    classifier = get_classifier(parsed_args)
    predicted = classifier.predict(test_data)
    print(numpy.mean(predicted == test_data.target))
