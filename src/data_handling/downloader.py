from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import matplotlib.pyplot as pyplot

from src.data_handling.types.news_groups_wrapper import NewsGroupsWrapper
from src.data_handling.types.news_groups_data_enum import NewsGroupsDataEnum


def get_data(sample_size: int, show_distribution: bool) -> NewsGroupsWrapper:
    news_groups_raw = fetch_20newsgroups(subset='all')

    news_groups_data = pd.DataFrame(
        {NewsGroupsDataEnum.DATA: news_groups_raw.data,
         NewsGroupsDataEnum.TARGET: news_groups_raw.target}
        )

    if (sample_size > 0):
        news_groups_data = news_groups_data.sample(sample_size)

    new_groups_wrapper = NewsGroupsWrapper(news_groups_data, news_groups_raw.target_names)

    # verify if distribution requires special handling
    value_probabilities = new_groups_wrapper.data[NewsGroupsDataEnum.TARGET].value_counts(normalize=True)
    value_probabilities = value_probabilities.sort_index()

    print('Data distribution')
    print(value_probabilities.rename(lambda category_id: new_groups_wrapper.group_names[category_id]))

    if (show_distribution):
        new_groups_wrapper.data.groupby(NewsGroupsDataEnum.TARGET).count().plot.bar()
        pyplot.show()

    return new_groups_wrapper
