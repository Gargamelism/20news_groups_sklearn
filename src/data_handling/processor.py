import re
import nltk
import pandas as pd
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Set, Any

from src.data_handling.types.data_cleaning_enum import DataCleaningEnum
from src.data_handling.types.vectorizer_conf import VectorizerConf
from src.ml_modeling.types.model_data_wrapper import ModelDataWrapper, ModelSplitWrapper


def remove_stopwords(text: str) -> str:
    split_text = text.split()

    stopwords = []
    # nltk stopwords needs to be downloaded first time it runs
    GET_STOPWORDS_RETRY_COUNT = 2
    for _ in range(GET_STOPWORDS_RETRY_COUNT):
        try:
            stopwords = nltk_stopwords.words('english')
            break
        except:
            nltk.download('stopwords')

    # Stopwords removal
    split_text = [word for word in split_text if (word not in stopwords)]

    return ' '.join(split_text)


def stem_text(text: str) -> str:
    split_text = text.split()

    stemmer = SnowballStemmer('english')
    split_text = [stemmer.stem(word) for word in split_text]  # stemming

    return ' '.join(split_text)


def clean(text: str, clean_methods: Set[DataCleaningEnum]) -> str:
    do_all = DataCleaningEnum.ALL in clean_methods

    if (do_all or DataCleaningEnum.SYMBOL_REMOVE in clean_methods):
        # Remove symbols (?!., etc.) - keep only words and numbers
        text = re.sub('[^a-zA-Z0-9]', ' ', text)

    if (do_all or DataCleaningEnum.NUMBER_INDICATE in clean_methods):
        # If number, indicate a number
        text = re.sub('[0-9]+', 'number', text)

    if (do_all or DataCleaningEnum.ALIGN_CASE in clean_methods):
        text = text.lower()

    if (do_all or DataCleaningEnum.REMOVE_STOPWORDS in clean_methods):
        text = remove_stopwords(text)

    if (do_all or DataCleaningEnum.STEM in clean_methods):
        text = stem_text(text)

    return text


def vectorizer_factory(data: ModelSplitWrapper, conf: VectorizerConf) -> Any:
    tfidf_vectorizer = TfidfVectorizer(
        ngram_range=(conf.ngram_range_start, conf.ngram_range_end), max_features=conf.max_features
        )

    tfidf_vectorizer.fit(data.data)

    return tfidf_vectorizer

def vectorize(data: ModelSplitWrapper, vectorizer: Any) -> ModelDataWrapper:
    vectorized_data_arr = vectorizer.transform(data.data).toarray()

    vectorized_data_df = pd.DataFrame(data=vectorized_data_arr, columns=vectorizer.get_feature_names_out())

    return ModelDataWrapper(vectorized_data_df, data.target)
