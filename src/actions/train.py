from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from dataclasses import dataclass
import typing


@dataclass
class ClassifierWrapper:
    classifier: typing.Any
    data_transformer: typing.Any


def get_classifier_func(classifier_name):
    return {"NaiveBayes": naive_bayes_classifier,
            "LogisticRegression": logistic_regression_classifier
            }.get(classifier_name)


def naive_bayes_classifier(pased_args, training_dataset, training_target) -> ClassifierWrapper:
    count_vectorizer = CountVectorizer()
    fit_transformed_data = count_vectorizer.fit_transform(training_dataset)

    tfidf_transformer = TfidfTransformer()
    tfidfed_fit_transformed_data = tfidf_transformer.fit_transform(fit_transformed_data)

    data_transformer = lambda data: tfidf_transformer.transform(count_vectorizer.transform(data))

    classifier = MultinomialNB()
    classifier.fit(tfidfed_fit_transformed_data, training_target)

    classifier_wrapper = ClassifierWrapper(classifier, data_transformer)
    return classifier_wrapper


def logistic_regression_classifier(parsed_args, training_dataset, training_target) -> ClassifierWrapper:
    count_vectorizer = CountVectorizer()
    fit_transformed_data = count_vectorizer.fit_transform(training_dataset)

    tfidf_transformer = TfidfTransformer()
    tfidfed_fit_transformed_data = tfidf_transformer.fit_transform(fit_transformed_data)

    data_transformer = lambda data: tfidf_transformer.transform(count_vectorizer.transform(data))

    classifier = LogisticRegression()
    classifier.fit(tfidfed_fit_transformed_data, training_target)

    classifier_wrapper = ClassifierWrapper(classifier, data_transformer)
    return classifier_wrapper


def get_classifier(parsed_args) -> ClassifierWrapper:
    dataset = fetch_20newsgroups()

    classifier_wrapper = get_classifier_func(parsed_args.classifier)(
        parsed_args, dataset.data, dataset.target
    )

    return classifier_wrapper
