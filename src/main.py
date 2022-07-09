import argparse
import sys
from typing import List

from actions.download import download_dataset
from actions.predict import predict


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

    train = subparsers.add_parser(name="train", help="train on dataset")
    train.set_defaults(func=predict)
    train.add_argument("-c", "--classifier", required=True, choices=["NaiveBayes", "LogisticRegression"])

    return parser


def main(args: List[str]):
    parser = parse_args()
    parsed_args = parser.parse_args()
    if(not hasattr(parsed_args, "func")):
        parser.print_usage()
        return

    parsed_args.func(parsed_args)


if __name__ == "__main__":
    main(sys.argv)
