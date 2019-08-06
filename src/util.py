import pandas as pd


def create_df():
    """ Generates the DataFrame for the course

     Returns:
         DataFrame: Contains information about vehicles
     """
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

    data_frame = pd.read_csv(url, header=None)
    data_frame.columns = column_names
    return data_frame


example_df = create_df()
# Print head/tail
example_df.head(5)
example_df.tail(5)

# Save DataFrame as a csv
example_df.to_csv('file_path.csv')

# Provide a statistical summary of everything
example_df.describe()

# Include non-numerical objects
example_df.describe(include='all')
