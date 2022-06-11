from sklearn.feature_extraction.text import TfidfVectorizer

class ManipulateData:
    @staticmethod
    def text_to_vectors(data):
        vectorizer = TfidfVectorizer()
        return vectorizer.fit_transform(data)

