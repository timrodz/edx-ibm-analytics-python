"""
DATA WRANGLING

- Handling missing values
- Converting categorial data -> numerical
"""
import pandas as pd
import numpy as np
import src.util as util

df = util.create_df()
"""
Operations
"""
# Show the shape of the dataset
df.shape()

# Show the data types for the data frame
df.dtypes()

# Highlights last 5 values and shows the data type
df['price'].tail()

# Converting a column's data type
df['price'] = df['price'].astype(int)

# Convert the column type to float/int if possible
df['normalized-losses'] = df['normalized-losses'].astype('float')

# Convert an entire entire column's values
df['city-mpg'] = 235 / df['city-mpg']  # mpg -> L/100Km

# Rename columns
df.rename(columns={'"highway-mpg"': 'highway-L/100km'}, inplace=True)

""" Dealing with missing values
1) Numerical values: replace these values with the mean
2) Categorical values: use the most common entry
3) Replace based on other functions
4) Leave it as missing data

-- NOTES
Whole columns should be dropped only if most entries in the column are empty.
"""
# Replace values with the mean
mean = df['normalized-losses'].mean(skipna=True)
df['normalized-losses'].fillna(mean, inplace=True)

# Remove NaN values in a string
df.replace(['NaN', 'NaT', '?'], np.nan, inplace=True)
df.dropna(subset=['price'], axis=0, how='all', inplace=True)

# Replace NaN values with 0 - The column must contain numerical values
df.fillna(0, inplace=True)

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
pd.get_dummies(df['fuel'])
