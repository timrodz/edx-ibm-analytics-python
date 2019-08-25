"""
EXPLORATORY DATA ANALYSIS
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

import src.util as util

df = util.create_df()
""" Overview

1) Descriptive Statistics
2) GroupBy
3) ANOVA — Analysis of variance
4) Correlation
5) Statistical Correlation
"""


def descriptive_statistics():
    # Descriptive Statistics
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
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.title('Scatterplot of Engine Size vs Price')
    plt.xlabel('Engine Size')
    plt.ylabel('Price')
    plt.show()


def group_by():
    """
    Group By

    Used on categorical variables (size, price, etc.). Groups data into subsets according to the different categories of the variable.

    Can be done on single or multiple variables.
    """
    df_test = df[['drive-wheels', 'body-style', 'price']]
    df_grp = df_test.groupby(['drive-wheels', 'body-style'], as_index=False).mean()
    
    """
    Pivot table & Heatmaps
    
    One variable displayed along the columns and the other variable displayed along the rows
    """
    df_pivot = df_grp.pivot(index='drive-wheels', columns='body-style')
    
    plt.pcolor(df_pivot, cmap='RdBu')
    plt.colorbar()
    plt.show()


def analysis_of_variance():
    """
    ANOVA
    
    Why? To find the correlation between different groups of a categorical variable
    
    What do we get from ANOVA?
    F-test score: variation between sample group means divided by variation within sample group
    P-value: confidence degree
    
    Small F implies poor correlation between the variable categories and the target variable.
    Large F implies strong correlation
    """
    df_anova = df[['make', 'price']]
    grouped_anova = df_anova.groupby(['make'])
    anova_results_1 = stats.f_oneway(grouped_anova.get_group('honda')['price'],
                                     grouped_anova.get_group('subaru')['price'])
    anova_results_2 = stats.f_oneway(grouped_anova.get_group('honda')['price'],
                                     grouped_anova.get_group('jaguar')['price'])
    print(anova_results_1)
    print(anova_results_2)


def correlation():
    """
    Statistical metric for measuring interdependency of 2 variables
    
    Measures to what extent different variables are interdependent
    
    Examples:
        Lung cancer -> Smoking
        Rain -> Umbrella
    
    Correlation does not imply causation
    The umbrella didn't cause the rain, and the rain didn't cause the umbrella
    """
    # Positive Linear Relationship
    sns.regplot(x='engine-size', y='price', data=df)
    plt.ylim(0,)


if __name__ == "__main__":
    correlation()
