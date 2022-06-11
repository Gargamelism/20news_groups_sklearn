import sklearn.datasets

def download_dataset(parsed_args):
    return sklearn.datasets.fetch_20newsgroups(subset="all")