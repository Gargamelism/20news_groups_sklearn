from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def get_classifier_func(classifier_name):
    return {"NaiveBayes": naive_bayes_classifier}.get(classifier_name)


def naive_bayes_classifier(pased_args, training_dataset, training_target):
    classifier = Pipeline(
            [
                ("vect", CountVectorizer()),
                ("tfidf", TfidfTransformer()),
                ("classifier", MultinomialNB())
            ]
        )
    return classifier.fit(training_dataset, training_target)


def get_classifier(parsed_args):
    dataset = fetch_20newsgroups()

    classifier = get_classifier_func(parsed_args.classifier)(
        parsed_args, dataset.data, dataset.target
    )

    return classifier
