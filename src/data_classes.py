from dataclasses import dataclass


@dataclass
class ParsedArgs:
    pre_processing: str
    data_format: str
    classifier: str
