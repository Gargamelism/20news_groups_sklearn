from data_classes import ParsedArgs
from helpers.functional import identity

def get_preprocess_func(preprocess_name: str):
    if not preprocess_name:
        return identity

    return {}.get(preprocess_name)

def preprocess_data(parsed_args: ParsedArgs, raw_data):
    preprocessor = get_preprocess_func(parsed_args.pre_processing)
    return preprocessor(raw_data)
