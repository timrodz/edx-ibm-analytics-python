"""
EXPLORATORY DATA ANALYSIS
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import src.util as util

df = util.create_df()
""" Overview

1) Descriptive Statistics
2) GroupBy
3) ANOVA — Analysis of variance
4) Correlation
5) Statistical Correlation
"""
# Generate various summary statistics, excluding NaN values
df.describe()

# Summarize categorical data
df['drive-wheels'].value_counts()

# Helps spot outliers in a data set
sns.boxplot(x='drive-wheels', y='price', data=df)
plt.show()

# Clear the current figure so it does not interfere with our new plot
plt.clf()

# Scatter plot shows the relationship between two variables
# Predictor/Independent variables on x-axis
# Target/Dependent variables on y-axis
y = df['engine-size']
x = df['price']
plt.scatter(x, y)
plt.ylim(ymin=0)
plt.xlim(xmin=0)
plt.title('Scatterplot of Engine Size vs Price')
plt.xlabel('Engine Size')
plt.ylabel('Price')
plt.show()
