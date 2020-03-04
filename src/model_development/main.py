"""
MODEL DEVELOPMENT

- Definition of a Model/Estimator
- Simple & Multiple Linear Regression
- Model Evaluation using Visualization
- Polynomial Regression and Pipelines
- R-squared and MSE for In-Sample Evaluation
- Prediction and Decision Making
"""
import matplotlib.pyplot as plt
from sklearn import linear_model
import seaborn as sns
from src import util

""" Model/Estimator
- Mathematical equation used to predict a value:
    - given 1 or more values
    - relating 1 or more PIV to TDV

Example:
- Input: A car model's highway miles per gallon (MPG) as the PIV
- TDV: The car's price

Formula: PIV -> Model -> TDV
Values:  55mpg -> Model -> $5000
Explained: We predict car with 55mpg will cost $5000.

Important:
More relevant data can lead to higher model accuracy. This is not always the case.

PIV: x axis
TDV: y axis - Will always contain the predicted values
"""


def simple_linear_regression():
    """
    Will only use 1 PIV to make 1 prediction (TDV)
    
    FORMULA: y = b0 + (b1 * x)
    b0: intercept
    b1: slope
    """
    from sklearn.metrics import mean_squared_error, r2_score

    df = util.create_df()

    # Define PIV and TDV
    x = df[['highway-mpg']]
    y = df['price']

    x_train = x[:-20]
    x_test = x[-20:]

    y_train = y[:-20]
    y_test = y[-20:]

    lmr = linear_model.LinearRegression()

    # Train/Fit the model
    lmr.fit(x_train, y_train)

    y_predict = lmr.predict(x_test)

    # The coefficients
    print('Coefficients: \n', lmr.coef_)
    # The mean squared error
    print('Mean squared error: %.2f'
          % mean_squared_error(y_test, y_predict))
    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination: %.2f'
          % r2_score(y_test, y_predict))

    width, height = 12, 10
    plt.figure(figsize=(width, height))
    plt.scatter(x_test, y_test, color='black')
    plt.plot(x_test, y_predict, color='blue', linewidth=3)
    plt.ylim(0,)
    plt.title('SLR model for predicting price')
    plt.xlabel('Miles Per Gallon')
    plt.ylabel('Price')
    plt.show()


def multiple_linear_regression():
    """
    Will use 2+ PIVs to make 1 prediction (TDV)
    """
    df = util.create_df()

    x = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
    y = df['price']

    lmr = linear_model.LinearRegression()

    lmr.fit(x, y)

    y_hat = lmr.predict(x)

    width, height = 12, 10
    plt.figure(figsize=(width, height))

    ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
    sns.distplot(y_hat, hist=False, color="b", label="Fitted Values", ax=ax1)

    plt.title('Actual vs Fitted Values for Price')
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')

    plt.show()


def polynomial_regression():
    """
    Special case of the general linear regression model
    Useful for describing 'curvilinear' relationships: This is what you get by squaring or setting
        higher-order terms of the predictor variables

    The model can be:
    - Quadratic (2nd order)
    - Cubic (3rd order)
    - Higher order (4th order +)

    The degree of the regression can make a big difference if you pick the right value
    .
    """
    import numpy as np
    from sklearn.preprocessing import PolynomialFeatures
    df = util.create_df()

    x = df['horsepower']
    y = df['curb-weight']

    f = np.polyfit(x, y, 3)
    p = np.poly1d(f)
    print(p)

    pr = PolynomialFeatures(degree=2, include_bias=False)
    x_poly = pr.fit_transform(df[['horsepower', 'curb-weight']])

    print(x_poly)


def model_evaluation_using_visualization():
    """
    Regression plot
        - Gives us a good estimate of:
            1) The relationship between 2 variables
            2) The strength of the correlation (r2)
            3) The direction of the relationship (Positive/Negative)

        - Combination of:
            1) Scatter plot: Every point represents a different y
            2) Fitted linear regression line

    Residual Plot
        - Represents the error between the actual values
        - y axis: residuals
        - x axis: TDV / Fitted values
        - Obtain difference by subtracting the predicted value - TDV
        - We expect results to have zero mean (Small variance), distributed evenly
            around the x axis with similar variance.
        - If there is NO curvature, a linear plot (function) might be more appropriate.
        - If there is a curvature, our LINEAR ASSUMPTION is incorrect, and it suggests
            a non-linear function
        - If the variance of the residuals increases with x, our MODEL is incorrect.

    Distribution Plot
        - Counts the predicted vs. actual values
        - Very useful for visualizing models with more than PIV

        Example
            - Given a data set of y values: 1, 2, 3
            - Count and plot the number of predicted values and TDVs
                that are approximately equal to 1, 2 and 3
            -
    """
    import seaborn as sns

    df = util.create_df()

    # Regression plot
    sns.regplot(x='highway-mpg', y='price', data=df)
    plt.ylim(0, )
    plt.show()

    plt.clf()

    # Residual plot
    sns.residplot(df['highway-mpg'], df['price'])
    plt.ylim(0, )
    plt.show()

    plt.clf()

    # Distribution plot
    x = df[['highway-mpg']]
    y = df['price']

    lmr = linear_model.LinearRegression()

    lmr.fit(x, y)

    y_hat = lmr.predict(x)

    ax1 = sns.distplot(y, hist=False, color='r', label='Actual value')
    sns.distplot(y_hat, hist=False, color='b', label='Fitted values', ax=ax1)
    plt.show()


def calculate_mean_squared_error():
    """
    As the MSE increases, the prediction will be less accurate.
    """
    from sklearn.metrics import mean_squared_error
    df = util.create_df()

    x = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
    y = df['price']

    lmr = linear_model.LinearRegression()

    lmr.fit(x, y)

    y_hat = lmr.predict(x)
    mse = mean_squared_error(df['price'], y_hat)
    print(mse)


if __name__ == "__main__":
    simple_linear_regression()
    multiple_linear_regression()
    # polynomial_regression()
    # model_evaluation_using_visualization()
    # calculate_mean_squared_error()
