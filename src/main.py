import argparse

from ParsedArgs import ParsedArgs
import data_handling.downloader as downloader

def parse_args():
    parser = argparse.ArgumentParser(
        description="parser for exploration of scikit groups text"
    )

    parser.add_argument('-s', '--sample-size', type=int, default=0, help='Limit data size')
    parser.add_argument('-d', '--show-target-distribution', type=bool, default=False, help='Show plot of target distribution')
    parser.add_argument('-p', '--data-processors', default=[], choices=[])
    parser.add_argument('-t', '--data-vectorizers', default=[], choices=[])
    parser.add_argument('-c', '--classifiers', default=[], choices=[])

    return parser.parse_args()


def main():
    parsed_args = parse_args()
    parsed_args = ParsedArgs(**vars(parsed_args))

    downloader.get_data(parsed_args.sample_size, parsed_args.show_target_distribution)

    print('Done!')



if __name__ == "__main__":
    main()
