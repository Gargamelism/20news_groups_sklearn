from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from data_classes import ParsedArgs
from helpers.functional import identity


def get_classifier_func(classifier_name):
    if not classifier_name:
        return identity

    return {"NaiveBayes": naive_bayes_classifier,
            "LogisticRegression": logistic_regression_classifier
            }.get(classifier_name)


def naive_bayes_classifier(pased_args: ParsedArgs, training_data, training_target):
    classifier = MultinomialNB()
    classifier.fit(training_data, training_target)

    return classifier


def logistic_regression_classifier(parsed_args, training_data, training_target):
    classifier = LogisticRegression()
    classifier.fit(training_data, training_target)

    return classifier


def get_trained_classifier(parsed_args: ParsedArgs, training_data, training_target):
    classifier_func = get_classifier_func(parsed_args.classifier)
    classifier = classifier_func(parsed_args, training_data, training_target)

    return classifier
