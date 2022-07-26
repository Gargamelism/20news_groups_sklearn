import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, roc_curve


class Modeling:
    @classmethod
    def build_model(cls, X_train, y_train):

        results = {}

        # Logistic regression
        classifier=LogisticRegression(random_state = 0)
        classifier.fit(X_train,y_train)
        results['Logistic Regrassion'] = classifier

        # Naive Bayes
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)
        results['Naive Bayes'] = classifier

        # Random forest
        classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
        classifier.fit(X_train, y_train)
        results['Random Forest'] = classifier

        # Xgboost
        classifier = XGBClassifier(random_state=42)
        classifier.fit(X_train, y_train)
        results['Xgboost'] = classifier

        # Feature importance RF
        impact = pd.DataFrame({'Feature': list(X_train), 'Importance': results['Random Forest'].feature_importances_})
        impact['Importance'] = impact['Importance'].round(decimals=5)
        impact = impact.sort_values(by=['Importance'], ascending=False).reset_index(drop=True)

        return results

    @classmethod
    def get_cutoff(cls, classifier, X_valid, y_valid):  # there's also an alternative method called youdens cutoff
        probs = classifier.predict_proba(X_valid)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_valid, probs)
        # j_scores = tpr-fpr
        # j_ordered = sorted(zip(j_scores,thresholds))
        # cutoff = j_ordered[-1][1]
        cutoff = thresholds[np.argmax(tpr - fpr)]
        return cutoff

    @classmethod
    def evaluate(cls, X_test, y_test, results, prob_baseline):

        # Baseline
        categories = list(prob_baseline.keys())
        probabilities = [prob_baseline[c] for c in categories]
        y_baseline = np.random.choice(categories, size=len(y_test), p=probabilities)
        print(classification_report(y_test, y_baseline))

        # Performance
        for name, model in results.items():
            print('\n', name)
            y_pred = model.predict(X_test)
            print(classification_report(y_test, y_pred))

            # y_pred = model.predict_proba(X_test)[:, 1]
            # cutoff = cls.get_cutoff(model, X_test, y_test)
            # y_pred = y_pred > cutoff
            # print('roc_auc', roc_auc_score(y_test, y_pred))
