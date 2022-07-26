from data_processing import DataPreprocessing
from modeling import Modeling
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    data = DataPreprocessing.get_data()
    data = data.sample(1000)
    data['clean_text'] = data['text'].apply(DataPreprocessing.clean)

    X, y = data['clean_text'], data['topic']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
    X_train, X_valid, X_test = DataPreprocessing.features(X_train, X_valid, X_test)

    prob_baseline = dict(y_train.value_counts(normalize=True))
    results = Modeling.build_model(X_train, y_train)
    Modeling.evaluate(X_valid, y_valid, results, prob_baseline)
    
    print('done')