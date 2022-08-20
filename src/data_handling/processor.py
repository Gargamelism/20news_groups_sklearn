import re
import nltk
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import SnowballStemmer
from typing import Set

from src.data_handling.DataCleaningEnum import DataCleaningEnum


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
