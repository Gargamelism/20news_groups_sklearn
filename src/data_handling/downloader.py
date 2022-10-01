from sklearn.datasets import fetch_20newsgroups
import pandas as pd

import src.data_handling.helper as data_helper
from src.data_handling.types.news_groups_wrapper import NewsGroupsWrapper
from src.data_handling.types.news_groups_data_enum import NewsGroupsDataEnum


def get_data(sample_size: int, show_distribution: bool) -> NewsGroupsWrapper:
    news_groups_raw = fetch_20newsgroups(subset = 'all')

    news_groups_data = pd.DataFrame(
        {
            NewsGroupsDataEnum.DATA: news_groups_raw.data,
            NewsGroupsDataEnum.TARGET: news_groups_raw.target
            }
        )

    if (sample_size > 0):
        news_groups_data = news_groups_data.sample(sample_size)

    new_groups_wrapper = NewsGroupsWrapper(news_groups_data, news_groups_raw.target_names, news_groups_data.copy(deep = True))

    data_helper.print_distribution(new_groups_wrapper.data[NewsGroupsDataEnum.TARGET], show_distribution)

    return new_groups_wrapper
