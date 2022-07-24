from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import dataclasses
from typing import Callable, List, Any

from data_classes import ParsedArgs
from helpers.functional import identity

@dataclasses.dataclass
class DataFormatterWrapper:
    count_vectorizer: Any
    tfidf: Any
    format: Callable[[List[Any]], List[Any]]

def get_formatter_func(formatter_name: str) -> DataFormatterWrapper:
    if not formatter_name:
        return identity

    return {"tfidf": tfidf_formatter}.get(formatter_name)


def tfidf_formatter(data):
    count_vectorizer = CountVectorizer()
    vectorized_data = count_vectorizer.fit_transform(data)

    tfidf_transformer = TfidfTransformer()
    tfidf_transformer.fit(vectorized_data)

    data_formatter = lambda data: tfidf_transformer.transform(count_vectorizer.transform(data))
    return [count_vectorizer, tfidf_transformer, data_formatter]


def get_data_formatter_wrapper(parsed_args: ParsedArgs, data) -> DataFormatterWrapper:
    data_formatter_func = get_formatter_func(parsed_args.data_format)
    [count_vectorizer, tfidf, data_formatter] = data_formatter_func(data)
    data_formatter_wrapper = DataFormatterWrapper(count_vectorizer, tfidf, data_formatter)
    return data_formatter_wrapper
