import re
import pandas as pd
import matplotlib.pyplot as pyplot

email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')


def get_email(text: str) -> str:
    email = email_pattern.search(text)

    return email[0].lower() if email else ''


def get_emails(data: pd.Series) -> pd.Series:
    emails = data.tolist()
    emails = map(get_email, emails)
    emails = filter(None, emails)
    emails = set(emails)
    emails = sorted(emails)

    return pd.Series(emails)


def print_distribution(vals: pd.Series, show_plot = False, name = 'Data'):
    value_probabilities = vals.value_counts(normalize = True)
    value_probabilities = value_probabilities.sort_index()

    print(f'{name} distribution')
    print(value_probabilities)

    if show_plot:
        vals.value_counts().plot.bar()
        pyplot.show()
