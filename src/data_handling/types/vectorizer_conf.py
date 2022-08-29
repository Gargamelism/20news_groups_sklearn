from dataclasses import dataclass

NGRAM_RANGE_START_DEFAULT = 1
NGRAM_RANGE_END_DEFAULT = 2
MAX_FEATURES_DEFAULT = 500


@dataclass
class VectorizerConf:
    ngram_range_start: int = NGRAM_RANGE_START_DEFAULT
    ngram_range_end: int = NGRAM_RANGE_END_DEFAULT
    max_features: int = MAX_FEATURES_DEFAULT
