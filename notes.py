import pandas as pd
import numpy as np

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'

# 'https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.names'
column_names = ['symboling',
                'normalized-losses',
                'make',
                'fuel',
                'aspiration',
                'num-of-doors',
                'body-style',
                'drive-wheels',
                'engine-location',
                'wheel-base',
                'length',
                'width',
                'height',
                'curb-weight',
                'engine',
                'num-of-cylinders',
                'engine-size',
                'fuel-system',
                'bore',
                'stroke',
                'compression-ratio',
                'horsepower',
                'peak-rpm',
                'city-mpg',
                'highway-mpg',
                'price'
                ]

df = pd.read_csv(url, header=None)
df.columns = column_names
df.head(5)
df.tail(5)

# Save DataFrame as a csv
# df.to_csv('file_path.csv')

# Provide a statistical summary of everything
df.describe()

# Include non-numerical objects
df.describe(include='all')

'''
DATA WRANGLING
'''

''' Drop missing values
There are a few possible outcomes to do this:
1) Numerical value: replace these values with the mean.
2) Categorical values: use the most common entry.
3) Replace based on other functions (try to guess)
4) Leave it as missing data
'''
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

''' Normalization of data
1) Simple Feature Scaling
	xNew = xOld/xMax
2) Min-Max
	xNew = (xOld-xMin)/(xMax-xMin)
3) Z-score
	xNew = (xOld-m)/sd
	m: Average -> mean()
	sd: Standard Deviation -> std()
'''
# Simple Feature Scaling
df['length'] = df['length'] / df['length'].max()

# Min-Max
df['length'] = (df['length'] - df['length'].min()) / \
    (df['length'].max()-df['length'].min())

# Z-score
df['length'] = (df['length'] - df['length'].mean())/df['length'].std()

''' Binning
Grouping values into bins
Converts numeric into categorical variables
Group a set of numerical values into a set of bins

Sometimes this can improve the accuracy of the data.

	pandas.cut()
'''
bins = np.linspace(min(df['price']), max(df['price']), 4)
group_names = ['Low', 'Medium', 'High']
df['price-binned'] = pd.cut(df['price'], bins,
                            labels=group_names, include_lowest=True)

''' Categorical -> Numeric
Solution: Add dummy variables for each unique category
Assign 0 or 1 in each category

e.g.
fuel | column, type: object
--- Entries
gas			0
diesel		1

1) One-hot encoding
	pandas.get_dummies()
'''
pd.get_dummies(df['fuel'])
