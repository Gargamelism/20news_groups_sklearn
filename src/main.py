import argparse
import sys
from typing import List

from actions.download import download_dataset
from actions.test import test


def parse_args():
    parser = argparse.ArgumentParser(
        description="parser for exploration of scikit groups text"
    )
    subparsers = parser.add_subparsers()

    download = subparsers.add_parser(
        name="download",
        help="download the sklearn 20 newsgroups text dataset",
    )
    download.set_defaults(func=download_dataset)

    test_subparser = subparsers.add_parser(name="test", help="test data transformers and classifiers on dataset")
    test_subparser.set_defaults(func=test)
    test_subparser.add_argument("-p", "--pre-processing", required=False)
    test_subparser.add_argument("-f", "--data-format", required=True, choices=["tfidf"])
    test_subparser.add_argument("-c", "--classifier", required=True, choices=["NaiveBayes", "LogisticRegression"])

    return parser


def main(args: List[str]):
    parser = parse_args()
    parsed_args = parser.parse_args()

    if (not hasattr(parsed_args, "func")):
        parser.print_usage()
        return

    parsed_args.func(parsed_args)


if __name__ == "__main__":
    main(sys.argv)
