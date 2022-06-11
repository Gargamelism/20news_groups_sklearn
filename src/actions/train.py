from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from helpers.ManipulateData import ManipulateData


def get_classifier_func(classifier_name):
    return {"NaiveBayes": naive_bayes_classifier}.get(classifier_name)


def naive_bayes_classifier(pased_args, training_dataset, training_target):
    return MultinomialNB().fit(training_dataset, training_target)


def get_classifier(parsed_args):
    dataset = fetch_20newsgroups()

    vectors = ManipulateData.text_to_vectors(dataset.data)
    classifier = get_classifier_func(parsed_args.classifier)(
        parsed_args, vectors, dataset.target
    )

    return classifier
