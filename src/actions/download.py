import sklearn.datasets

from data_classes import ParsedArgs

def download_dataset(parsed_args: ParsedArgs):
    return sklearn.datasets.fetch_20newsgroups(subset='all')