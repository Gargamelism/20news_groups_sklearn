from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

from data_classes import ParsedArgs
from helpers.functional import identity


def get_formatter_func(formatter_name: str):
    if not formatter_name:
        return identity

    return {"tfidf": tfidf_formatter}.get(formatter_name)


def tfidf_formatter(data):
    count_vectorizer = CountVectorizer()
    vectorized_data = count_vectorizer.fit_transform(data)

    tfidf_transformer = TfidfTransformer()
    tfidf_transformer.fit(vectorized_data)

    data_formatter = lambda data: tfidf_transformer.transform(count_vectorizer.transform(data))
    return data_formatter


def get_data_formatter(parsed_args: ParsedArgs, data):
    data_formatter = get_formatter_func(parsed_args.data_format)
    return data_formatter(data)
