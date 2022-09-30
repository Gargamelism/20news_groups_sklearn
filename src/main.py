import argparse
import pandas as pd
from datetime import datetime

from parsed_args import ParsedArgs
import data_handling.downloader as downloader
import data_handling.processor as processor
import data_handling.data_splitter as data_splitter
import data_handling.helper as data_helper
import ml_modeling.evaluator as evaluator
from data_handling.types.split_conf import SplitConf, SplitByEnum
from data_handling.types.news_groups_data_enum import NewsGroupsDataEnum
from data_handling.types.data_cleaning_enum import DataCleaningEnum
from data_handling.types.vectorizer_conf import VectorizerConf
from ml_modeling.types.model_classifier_enum import ModelClassifierEnum
from ml_modeling.classifier import classifier_factory
from src.ml_modeling.types.model_data_wrapper import ModelSplitWrapper


def parse_args():
    parser = argparse.ArgumentParser(
        description = 'parser for exploration of scikit groups text'
        )

    parser.add_argument(
        '-p', '--data-processors', default = [DataCleaningEnum.ALL.name], nargs = '*', choices = [cleaning.name.lower()
                                                                                                  for
                                                                                                  cleaning in
                                                                                                  DataCleaningEnum]
        )
    parser.add_argument('-t', '--data-vectorizers', default = [], choices = [])
    parser.add_argument('-c', '--classifiers', default = [], choices = [])

    parser.add_argument('-s', '--sample-size', type = int, default = 0, help = 'Limit data size')

    parser.add_argument('--split-validation', type = bool, default = False)
    parser.add_argument(
        '--show-target-distribution', type = bool, default = False, help = 'Show plot of target distribution'
        )
    parser.add_argument('--print-feature-importance', type = bool, default = False)
    parser.add_argument(
        '--separate-train-test', default = SplitByEnum.DEFAULT, choices = [split.name.lower()
                                                                           for
                                                                           split in
                                                                           SplitByEnum]
        )

    return parser.parse_args()


def main():
    parsed_args = parse_args()
    parsed_args = ParsedArgs(**vars(parsed_args))

    news_groups_data = downloader.get_data(parsed_args.sample_size, parsed_args.show_target_distribution)

    news_groups_data.data[NewsGroupsDataEnum.DATA] = news_groups_data.data[NewsGroupsDataEnum.DATA].map(
        lambda text: processor.clean(text, parsed_args.data_processors)
        )

    split_conf = SplitConf()
    if parsed_args.separate_train_test == SplitByEnum.EMAILS:
        emails = data_helper.get_emails(news_groups_data.orig_data[NewsGroupsDataEnum.DATA]).map(
            lambda email: processor.clean(email, parsed_args.data_processors)
            )
        split_conf.split_by = SplitByEnum.EMAILS
        split_conf.split_by_vals = emails

    train_data_split_wrapper, test_data_split_wrapper = data_splitter.split_data(
        news_groups_data.data[NewsGroupsDataEnum.DATA],
        news_groups_data.data[NewsGroupsDataEnum.TARGET],
        split_conf
        )
    data_helper.print_distribution(train_data_split_wrapper.target, name = 'Train')

    if parsed_args.separate_train_test == SplitByEnum.EMAILS:
        train_list = train_data_split_wrapper.data.to_list()
        emails = filter(
            lambda email: next(
                (text for text in train_list if email in text), False
                ), split_conf.split_by_vals
            )
        emails = pd.Series(emails)
        split_conf.split_by_vals = emails

    validate_data_split_wrapper = ModelSplitWrapper(pd.Series([]), pd.Series([]))
    if (parsed_args.split_validation):
        train_data_split_wrapper, validate_data_split_wrapper = data_splitter.split_data(
            train_data_split_wrapper.data,
            train_data_split_wrapper.target,
            split_conf
            )

    vectorizer = processor.vectorizer_factory(train_data_split_wrapper, VectorizerConf())

    train_data_wrapper, test_data_wrapper = [
        processor.vectorize(data_split_wrapper, vectorizer) for data_split_wrapper in
        [train_data_split_wrapper, test_data_split_wrapper]
        ]

    print('baseline prediction - random choice')
    evaluator.evaluate_baseline(test_data_wrapper.target, news_groups_data.group_names)
    for classifier_type in [ModelClassifierEnum.LogisticRegression, ModelClassifierEnum.GaussianNB,
                            ModelClassifierEnum.RandomForest, ModelClassifierEnum.XGBClassifier]:
        classifier = classifier_factory(classifier_type, train_data_wrapper, {})
        evaluation_result = evaluator.evaluate(classifier, test_data_wrapper, news_groups_data.group_names)

        filename = f'evaluation-{datetime.now().strftime("%Y-%m-%dT%H-%I")}.log'
        with open(filename, 'a') as evaluation_file:
            evaluation_file.write(f'{str(classifier_type)} \n')

            accuracy = evaluation_result.get('accuracy')
            weighted_avg = evaluation_result.get('weighted avg')
            macro_avg = evaluation_result.get('macro avg')

            evaluation_file.write(
                f'accuracy: {str(accuracy)} \n'
                f'weighted avg: {weighted_avg} \n'
                f'macro avg: {macro_avg} \n\n'
                )

            if parsed_args.print_feature_importance and classifier_type == ModelClassifierEnum.RandomForest:
                evaluation_file.write(classifier.feature_importance.sort_values([2]).to_csv())

    print('Done!')


if __name__ == '__main__':
    main()
