"""
BINNING

- Generic binning
- Indicator Variable (Dummy)
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import src.util as util


def binning():
    """ Grouping values into bins
    Converts numeric into categorical variables
    Group a set of numerical values into a set of bins

    Sometimes this can improve the accuracy of the data
    """
    df = util.create_df()
    util.replace_nan_with_mean(df['horsepower'], 'float')
    df["horsepower"] = df["horsepower"].astype(int)

    # Return evenly spaced numbers over a specified interval - 4 dividers (3 bins)
    bins = np.linspace(min(df['horsepower']), max(df['horsepower']), 4)

    # Set names for our groups
    group_names = ['Low', 'Medium', 'High']

    # Bin values into discrete intervals
    # It runs through every value and applies the label depending on their ranges
    df['horsepower-binned'] = pd.cut(
        df['horsepower'],
        bins,
        labels=group_names,
        include_lowest=True
    )

    # Visualize it
    fig = plt.figure(figsize=(12, 14))
    plt.bar(group_names, df['horsepower-binned'].value_counts())
    fig.suptitle('Horsepower Bins', fontsize=18)
    plt.xlabel('Horsepower', fontsize=18)
    plt.ylabel('count', fontsize=16)
    fig.savefig('test.jpg')


def dummies():
    """ Turning categorical variables into quantitative variables
    Solution: Add dummy variables for each unique category
    Assign 0 or 1 in each category

    e.g.
    fuel | column, type: object
    --- Entries
    gas     0
    diesel  1

    1) One-hot encoding
      pandas.get_dummies()

    -- Indicator Variable
    An indicator variable (or dummy variable) is a numerical variable used to
    label categories. They are called 'dummies' because the numbers themselves
    don't have inherent meaning
    """
    df = util.create_df()
    dummy_var = pd.get_dummies(df['fuel'])

    # merge data frame "df" and "dummy_var"
    df = pd.concat([df, dummy_var], axis=1)

    # drop original column "fuel-type" from "df"
    df.drop("fuel-type", axis=1, inplace=True)


if __name__ == "__main__":
    binning()
    dummies()
