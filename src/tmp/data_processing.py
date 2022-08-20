import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
import re
import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


class DataPreprocessing:

    @classmethod
    def get_data(cls):

        # Read
        data = fetch_20newsgroups(subset='all')
        df = pd.DataFrame({'text': data.data, 'topic': data.target})

        # Distribution / balancy
        print(df.shape)
        print(df['topic'].value_counts(normalize=True))
        # df.groupby('topic').count().plot.bar()
        # plt.show()

        return df

    @classmethod
    def clean(cls, text, stemming=True):

        # Remove symbols (?!., etc.) - keep only words and numbers
        text = re.sub('[^a-zA-Z0-9]', ' ', text)
        # If number, indicate a number
        text = re.sub('[0-9]+', 'number', text)
        # Uncapitalize
        text = text.lower()

        # Split
        text = text.split()

        # nltk stopwords needs to be downloaded first time it runs
        for retry in range(2):
            try:
                stop = stopwords.words('english')
            except:
                nltk.download('stopwords')

        # Stopwords removal
        text = [word for word in text if (word not in set(stop))]
        # Stem
        if stemming:
            snowball = SnowballStemmer('english')
            text = [snowball.stem(word) for word in text]  # stemming

        # Return a list of words
        text = ' '.join(text)
        return text

    @classmethod
    def features(cls, X_train, X_valid, X_test):

        # Vectorizer
        vect = TfidfVectorizer(ngram_range=(1, 2), max_features=500)

        # Fit and transform
        X_train = vect.fit_transform(X_train).toarray()
        X_valid = vect.transform(X_valid).toarray()
        X_test = vect.transform(X_test).toarray()

        # Feature names
        X_train = pd.DataFrame(data=X_train, columns=vect.get_feature_names_out())
        X_valid = pd.DataFrame(data=X_valid, columns=vect.get_feature_names_out())
        X_test = pd.DataFrame(data=X_test, columns=vect.get_feature_names_out())

        return X_train, X_valid, X_test
