"""
DATA WRANGLING

- Handling missing values
- Converting categorial data -> numerical
"""
import numpy as np

from src import util

df = util.create_df()


def basic_operations():
    # Show the shape of the dataset
    shape = df.shape

    print(f'Shape: {shape}')

    # Show the data types for the data frame
    types = df.dtypes
    print(f'Types: {types}')

    # Highlights last 5 values and shows the data type
    df['price'].tail()

    # Converting a column's data type
    df['price'] = df['price'].astype(int)

    # Convert the column type to float/int if possible
    df['normalized-losses'] = df['normalized-losses'].astype('float')


def missing_values():
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


def standardization():
    """ Data standardization
    Standardization is the process of transforming data into a common format 
    which allows the researcher to make the meaningful comparison. 
    """
    # Convert an entire entire column's values
    df['city-mpg'] = 235 / df['city-mpg']  # mpg -> L/100Km

    # Rename columns
    df.rename(columns={'"highway-mpg"': 'highway-L/100km'}, inplace=True)


if __name__ == "__main__":
    basic_operations()
    missing_values()
    standardization()
