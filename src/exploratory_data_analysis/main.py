"""
EXPLORATORY DATA ANALYSIS

- Descriptive Statistics
- GroupBy
- ANOVA — Analysis of variance
- Correlation
- Statistical Correlation
"""
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from src import util


def descriptive_statistics():
    df = util.create_df()
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
    # PIV variables on x-axis
    # TDV variables on y-axis
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

    Used on categorical variables (size, price, etc.). Groups data into subsets
    according to the different categories of the variable.

    Can be done on single or multiple variables.
    """
    df = util.create_df()

    df_test = df[['drive-wheels', 'body-style', 'price']]
    df_grp = df_test.groupby(['drive-wheels', 'body-style'],
                             as_index=False).mean()

    """
    Pivot table & Heatmaps
    
    One variable displayed along the columns and the other variable displayed along the 
    rows
    """
    df_pivot = df_grp.pivot(index='drive-wheels', columns='body-style')

    plt.pcolor(df_pivot, cmap='RdBu')
    plt.colorbar()
    plt.show()


def analysis_of_variance():
    """
    ANOVA

    Why? To find the correlation between different
    groups of a categorical variable

    What do we get from ANOVA?
    F-test score: variation between sample group means divided
        by variation within sample group
    P-value: confidence degree

    Notes:
    - Small F implies poor correlation between the variable
        categories and the target variable.
    - Large F implies strong correlation
    """
    df = util.create_df()

    df_anova = df[['make', 'price']]
    grouped_anova = df_anova.groupby(['make'])
    anova_results_1 = stats.f_oneway(grouped_anova.get_group('honda')['price'],
                                     grouped_anova.get_group('subaru')['price'])
    anova_results_2 = stats.f_oneway(grouped_anova.get_group('honda')['price'],
                                     grouped_anova.get_group('jaguar')['price'])
    print(anova_results_1)
    print(anova_results_2)


def correlation_simple():
    """
    Statistical metric for measuring interdependency of 2 variables

    Measures to what extent different variables are interdependent

    Examples:
        Lung cancer -> Smoking
        Rain -> Umbrella

    Correlation does not imply causation
    The umbrella didn't cause the rain, and the rain didn't cause the umbrella
    """
    df = util.create_df()

    # Positive Linear Relationship
    sns.regplot(x='engine-size', y='price', data=df)
    plt.ylim(0, )
    plt.show()

    plt.clf()

    # Negative Linear Relationship
    sns.regplot('highway-mpg', 'price', data=df)
    plt.ylim(0, )
    plt.show()

    plt.clf()

    # Weak Linear Relationship
    sns.regplot('peak-rpm', 'price', data=df)
    plt.ylim(0, )
    plt.show()


def correlation_statistics():
    """
    Pearson correlation
        - Correlation Coefficient. Explanation:
            - Close to +1: Large positive relationship
            - Close to -1: Large negative relationship
            - Close to 0: No relationship

        - P Value. Strength of result certainty:
            - <0.001: Strong certainty
            - <0.05: Moderate certainty
            - <0.1: Weak certainty
            - >0.1: No certainty

    Notes:
        https://en.wikipedia.org/wiki/Correlation_and_dependence
        - We can say there's a strong correlation when:
            1. Correlation Coefficient is close to 1 or -1
            2. P value is less than 0.001
        - If the correlation coefficient is NaN?
    """
    df = util.create_df()

    df['horsepower'] = df['horsepower'].astype(float)
    pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])
    print('Coef: {} | P value: {}'.format(pearson_coef, p_value))


if __name__ == "__main__":
    descriptive_statistics()
    group_by()
    analysis_of_variance()
    correlation_simple()
    correlation_statistics()
