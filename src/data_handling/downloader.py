from sklearn.datasets import fetch_20newsgroups
import pandas
import matplotlib.pyplot as pyplot

from enum import Enum
from dataclasses import dataclass
from typing import List


class NewsGroupsDataEnum(Enum):
    DATA = 'DATA'
    TARGET = 'TARGET'


@dataclass
class NewsGroupsWrapper:
    data: pandas.DataFrame
    group_names: List[str]


def get_data(sample_size: int, show_distribution: bool) -> NewsGroupsWrapper:
    news_groups_raw = fetch_20newsgroups(subset="all")

    news_groups_data = pandas.DataFrame(
        {NewsGroupsDataEnum.DATA: news_groups_raw.data, NewsGroupsDataEnum.TARGET: news_groups_raw.target})

    if (sample_size > 0):
        news_groups_data = news_groups_data.sample(sample_size)

    new_groups_wrapper = NewsGroupsWrapper(news_groups_data, news_groups_raw.target_names)

    # verify if distribution requires special handling
    print(new_groups_wrapper.data.shape)
    print(new_groups_wrapper.data[NewsGroupsDataEnum.TARGET].value_counts(normalize=True))

    if (show_distribution):
        new_groups_wrapper.data.groupby(NewsGroupsDataEnum.TARGET).count().plot.bar()
        pyplot.show()

    return new_groups_wrapper
