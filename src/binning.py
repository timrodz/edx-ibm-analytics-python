"""
BINNING
"""
import pandas as pd
import numpy as np
import src.util as util

df = util.create_df()
""" Binning
Grouping values into bins
Converts numeric into categorical variables
Group a set of numerical values into a set of bins

Sometimes this can improve the accuracy of the data.

	pandas.cut()
"""
bins = np.linspace(min(df['price']), max(df['price']), 4)
group_names = ['Low', 'Medium', 'High']
df['price-binned'] = pd.cut(df['price'], bins,
                            labels=group_names, include_lowest=True)
