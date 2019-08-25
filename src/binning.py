"""
BINNING
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import src.util as util

df = util.create_df()
""" Grouping values into bins
Converts numeric into categorical variables
Group a set of numerical values into a set of bins

Sometimes this can improve the accuracy of the data
"""


def bin():
    util.replace_with_mean(df['horsepower'], 'float')
    df["horsepower"] = df["horsepower"].astype(int)

    # Return evenly spaced numbers over a specified interval - 4 dividers (3 bins)
    bins = np.linspace(min(df['horsepower']), max(df['horsepower']), 4)

    group_names = ['Low', 'Medium', 'High']

    # Bin values into discrete intervals
    df['horsepower-binned'] = pd.cut(
        df['horsepower'],
        bins,
        labels=group_names,
        include_lowest=True
    )

    fig = plt.figure(figsize=(12, 14))
    plt.bar(group_names, df['horsepower-binned'].value_counts())
    fig.suptitle('Horsepower Bins', fontsize=18)
    plt.xlabel('Horsepower', fontsize=18)
    plt.ylabel('count', fontsize=16)
    fig.savefig('test.jpg')
