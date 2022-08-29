from enum import IntEnum


class ModelClassifierEnum(IntEnum):
    LogisticRegression = 1
    GaussianNB = 2
    RandomForest = 3
    XGBClassifier = 4

    def __str__(self):
        display_name = {
            self.LogisticRegression.value: 'LogisticRegression',
            self.GaussianNB.value: 'GaussianNB',
            self.RandomForest.value: 'RandomForest',
            self.XGBClassifier.value: 'XGBClassifier',
            }.get(self.value)

        return display_name
