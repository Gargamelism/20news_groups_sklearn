import argparse

from parsed_args import ParsedArgs
import data_handling.downloader as downloader
import data_handling.processor as processor
import data_handling.data_splitter as data_splitter
import ml_modeling.evaluator as evaluator
from data_handling.types.news_groups_data_enum import NewsGroupsDataEnum
from data_handling.types.data_cleaning_enum import DataCleaningEnum
from data_handling.types.vectorizer_conf import VectorizerConf
from ml_modeling.types.model_classifier_enum import ModelClassifierEnum
from ml_modeling.classifier import classifier_factory


def parse_args():
    parser = argparse.ArgumentParser(
        description='parser for exploration of scikit groups text'
        )

    parser.add_argument('-s', '--sample-size', type=int, default=0, help='Limit data size')
    parser.add_argument(
        '-d', '--show-target-distribution', type=bool, default=False, help='Show plot of target distribution'
        )
    parser.add_argument('-p', '--data-processors', default=[], choices=[])
    parser.add_argument('-t', '--data-vectorizers', default=[], choices=[])
    parser.add_argument('-c', '--classifiers', default=[], choices=[])

    return parser.parse_args()


def main():
    parsed_args = parse_args()
    parsed_args = ParsedArgs(**vars(parsed_args))

    news_groups_data = downloader.get_data(parsed_args.sample_size, parsed_args.show_target_distribution)
    news_groups_data.data[NewsGroupsDataEnum.DATA] = news_groups_data.data[NewsGroupsDataEnum.DATA].map(
        lambda text: processor.clean(text, {DataCleaningEnum.ALL})
        )

    train_data_split_wrapper, validate_data_split_wrapper, test_data_split_wrapper = data_splitter.split_data(
        news_groups_data.data[NewsGroupsDataEnum.DATA],
        news_groups_data.data[NewsGroupsDataEnum.TARGET]
        )

    vectorizer = processor.vectorizer_factory(train_data_split_wrapper, VectorizerConf())

    train_data_wrapper, validate_data_wrapper, test_data_wrapper = [
        processor.vectorize(data_split_wrapper, vectorizer)
        for data_split_wrapper in
        [train_data_split_wrapper, validate_data_split_wrapper,
         test_data_split_wrapper]]

    print('baseline prediction - random choice')
    evaluator.evaluate_baseline(test_data_wrapper.target, news_groups_data.group_names)
    for classifier_type in [ModelClassifierEnum.LogisticRegression, ModelClassifierEnum.GaussianNB,
                            ModelClassifierEnum.RandomForest, ModelClassifierEnum.XGBClassifier]:
        classifier = classifier_factory(classifier_type, train_data_wrapper, {})
        evaluator.evaluate(classifier, test_data_wrapper, news_groups_data.group_names)

    print('Done!')


if __name__ == '__main__':
    main()
