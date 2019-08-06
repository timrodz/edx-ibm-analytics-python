"""
DATA WRANGLING

- Handling missing values
- Converting categorial data -> numerical
"""
import pandas as pd
import numpy as np
import src.util as util

df = util.create_df()
""" Drop missing values
There are a few possible outcomes to do this:
1) Numerical value: replace these values with the mean.
2) Categorical values: use the most common entry.
3) Replace based on other functions (try to guess)
4) Leave it as missing data
"""
# Replace entries of these values in the string
df.replace(['NaN', 'NaT', '?'], np.nan, inplace=True)
df.dropna(subset=['price'], axis=0, how='all', inplace=True)

# Convert the column type to float/int if possible
df['normalized-losses'] = df['normalized-losses'].astype('float')

# Replace values with the mean
mean = df['normalized-losses'].mean(skipna=True)
df['normalized-losses'].fillna(mean, inplace=True)

# Convert entire column to L/100Km
df['city-mpg'] = 235 / df['city-mpg']

# Highlights last 5 values and shows the data type
df['price'].tail()

# Converts types to a specific one
df['price'] = df['price'].astype('int')

""" Categorical -> Numeric
Solution: Add dummy variables for each unique category
Assign 0 or 1 in each category

e.g.
fuel | column, type: object
--- Entries
gas     0
diesel  1

1) One-hot encoding
	pandas.get_dummies()
"""
pd.get_dummies(df['fuel'])
