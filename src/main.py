import argparse
import sys
from typing import List

from actions.download import download_dataset
from actions.predict import predict


def parse_args(args):
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
    train.add_argument("-c", "--classifier", required=True, choices=["NaiveBayes"])

    return parser.parse_args(args)


def main(args: List[str]):
    parsed_args = parse_args(args[1:])
    parsed_args.func(parsed_args)


if __name__ == "__main__":
    main(sys.argv)
