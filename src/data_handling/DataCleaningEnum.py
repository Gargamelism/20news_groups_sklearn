from enum import IntEnum, auto


class DataCleaningEnum(IntEnum):
    ALL = auto()
    SYMBOL_REMOVE = auto()
    NUMBER_INDICATE = auto()
    ALIGN_CASE = auto()
    REMOVE_STOPWORDS = auto()
    STEM = auto()
