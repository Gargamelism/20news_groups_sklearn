import argparse

from ParsedArgs import ParsedArgs
import data_handling.downloader as downloader
import data_handling.processor as processor
import data_handling.data_splitter as data_splitter
from data_handling.types.NewsGroupsDataEnum import NewsGroupsDataEnum
from data_handling.types.DataCleaningEnum import DataCleaningEnum


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

    train_data, validate_data, test_data = data_splitter.split_data(
        news_groups_data.data[NewsGroupsDataEnum.DATA],
        news_groups_data.data[NewsGroupsDataEnum.TARGET]
        )

    print('Done!')


if __name__ == '__main__':
    main()
