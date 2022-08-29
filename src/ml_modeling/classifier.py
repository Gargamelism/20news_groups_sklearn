import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from typing import Dict, Union

from src.ml_modeling.types.model_classifier_enum import ModelClassifierEnum
from src.ml_modeling.types.model_data_wrapper import ModelDataWrapper
from src.ml_modeling.types.classifier_wrapper import ClassifierWrapper, FeatureImportanceEnum
import src.ml_modeling.types.classifiers_consts as classifiers_consts


def logistic_regression_classifier_factory(data_wrapper: ModelDataWrapper, conf: Dict) -> ClassifierWrapper:
    random_state = conf.get('random_state', classifiers_consts.LOGISTIC_REGRESSION_RANDOM_STATE)

    logistic_regression = LogisticRegression(random_state=random_state, max_iter=1000)
    logistic_regression.fit(data_wrapper.data, data_wrapper.target)
    return ClassifierWrapper(ModelClassifierEnum.LogisticRegression, logistic_regression)


def gaussian_nb_classifier_factory(data_wrapper: ModelDataWrapper, conf: Dict) -> ClassifierWrapper:
    gaussian_nb = GaussianNB()
    gaussian_nb.fit(data_wrapper.data, data_wrapper.target)
    return ClassifierWrapper(ModelClassifierEnum.GaussianNB, gaussian_nb)


def random_forest_classifier_factory(data_wrapper: ModelDataWrapper, conf: Dict) -> ClassifierWrapper:
    n_estimators = conf.get('n_estimators', classifiers_consts.RANDOM_FOREST_N_ESTIMATORS)
    criterion = conf.get('criterion', classifiers_consts.RANDOM_FOREST_CRITERION)
    random_state = conf.get('random_state', classifiers_consts.RANDOM_FOREST_RANDOM_STATE)

    random_forest = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, random_state=random_state)
    random_forest.fit(data_wrapper.data, data_wrapper.target)

    feature_importance = pd.DataFrame({FeatureImportanceEnum.FEATURES: data_wrapper.data.columns.tolist(), FeatureImportanceEnum.IMPORTANCE: random_forest.feature_importances_})
    feature_importance[FeatureImportanceEnum.IMPORTANCE] = feature_importance[FeatureImportanceEnum.IMPORTANCE].round(decimals=5)
    feature_importance = feature_importance.sort_values(by=[FeatureImportanceEnum.IMPORTANCE], ascending=False).reset_index(drop=True)

    return ClassifierWrapper(ModelClassifierEnum.RandomForest, random_forest, feature_importance)


def xg_boost_classifier_factory(data_wrapper: ModelDataWrapper, conf: Dict) -> ClassifierWrapper:
    random_state = conf.get('random_state', classifiers_consts.XG_BOOST_RANDOM_STATE)

    xg_boost = XGBClassifier(random_state=random_state)
    xg_boost.fit(data_wrapper.data, data_wrapper.target)
    return ClassifierWrapper(ModelClassifierEnum.XGBClassifier, xg_boost)


def get_classifier_factory_func(type: ModelClassifierEnum):
    return {
        ModelClassifierEnum.LogisticRegression: logistic_regression_classifier_factory,
        ModelClassifierEnum.GaussianNB: gaussian_nb_classifier_factory,
        ModelClassifierEnum.RandomForest: random_forest_classifier_factory,
        ModelClassifierEnum.XGBClassifier: xg_boost_classifier_factory,
        }.get(type, None)


def classifier_factory(type: ModelClassifierEnum, data_wrapper: ModelDataWrapper, conf: Dict) -> Union[
    ClassifierWrapper, None]:
    classifier_factory_func = get_classifier_factory_func(type)
    if (not classifier_factory_func):
        print('what? what? what? try a different classifier!')
        return

    return classifier_factory_func(data_wrapper, conf)
