"""
MODEL EVALUATION

- How to evaluate models
- Over/Under-fitting
- Model Selection
- Ridge Regression
- Grid Search

Qs:
- How can you be certain your model works in the real world and performs optimally?
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from src import util
import matplotlib.pyplot as plt
import seaborn as sns


def model_evaluation():
    """
    Tells us how our model performs in the real world
    Difference with in-sample evaluation:
        - In-sample tells us how well our model fits the data already given to train it
        - Problem: It does not tell us how well the trained model can be used to predict new data
        - Solution: Split data in sets:
            - Training data: Train it with in-sample evaluation
            - Testing data:

    Example:
        - Train 70% of the data
        - Test 30% of the data

    There exists a generalization error that involves the percentages of data used for training
    and testing. TO overcome this issue, we use

    Cross Validation
        - Most common out-of-sample (testing) evaluation metric
        - More effective use of data (each observation is used for both training and testing)
    """
    from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
    from sklearn.linear_model import LinearRegression

    df = util.create_df()

    x_data = df[['highway-mpg']]
    y_data = df['price']

    lr = LinearRegression()
    lr.fit(x_data, y_data)
    # cv specifies how many folds to use
    scores = cross_val_score(lr, x_data, y_data, cv=3)
    print(f'Mean: {scores.mean()}. Standard deviation: {scores.std()}')
    predicted_scores = cross_val_predict(lr, x_data, y_data, cv=3)

    # Visualize the model
    width, height = 12, 10
    plt.figure(figsize=(width, height))
    sns.regplot(x='highway-mpg', y='price', data=df)
    plt.ylim(0, )
    plt.show()

    plt.clf()

    plt.figure(figsize=(width, height))
    sns.regplot(x="peak-rpm", y="price", data=df)
    plt.ylim(0, )
    plt.show()

    plt.clf()

    plt.figure(figsize=(width, height))
    sns.residplot(df['highway-mpg'], df['price'])
    plt.show()


def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    ax1 = sns.distplot(RedFunction, hist=False, color="r", label=RedName)
    ax2 = sns.distplot(BlueFunction, hist=False, color="b", label=BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')

    plt.show()
    plt.close()


def PollyPlot(xtrain, xtest, y_train, y_test, lr, poly_transform):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    # training data
    # testing data
    # lr:  linear regression object
    # poly_transform:  polynomial transformation object

    xmax = max([xtrain.values.max(), xtest.values.max()])

    xmin = min([xtrain.values.min(), xtest.values.min()])

    x = np.arange(xmin, xmax, 0.1)

    plt.plot(xtrain, y_train, 'ro', label='Training Data')
    plt.plot(xtest, y_test, 'go', label='Test Data')
    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))),
             label='Predicted Function')
    plt.ylim([-10000, 60000])
    plt.ylabel('Price')
    plt.legend()


def fitting():
    """
    Over/Under-fitting for polynomial regression
    How to pick the best polynomial order

    Under-fitting: the model is too simple to fit the data
    Over-fitting: The model is too flexible to fit the data.

    The training error decreases with the order of the polynomial, BUT
    The test error is a better means of estimating the error of a polynomial.
    """
    from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
    from sklearn.linear_model import LinearRegression

    df = util.create_df()

    # x_data = df[['highway-mpg']]
    x_data = df.drop('price', axis=1)
    y_data = df['price']
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                        test_size=0.15, random_state=1)

    print("number of test samples :", x_test.shape[0])
    print("number of training samples:", x_train.shape[0])

    lr = LinearRegression()
    lr.fit(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_train)

    yhat_train = lr.predict(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
    print(f'Train: {yhat_train[0:5]}')

    yhat_test = lr.predict(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
    print(f'Test: {yhat_test[0:5]}')

    title = 'Distribution Plot of Predicted Value Using Training Data vs Training Data Distribution'
    DistributionPlot(y_train, yhat_train, "Actual Values (Train)", "Predicted Values (Train)",
                     title)

    rsqu_test = []
    order = [1, 2, 3, 4]

    lr = LinearRegression()

    # Determine which polynomial degree gives the best r^2 value
    for n in order:
        pr = PolynomialFeatures(degree=n)
        x_train_pr = pr.fit_transform(x_train[['horsepower']])
        x_test_pr = pr.fit_transform(x_test[['horsepower']])
        lr.fit(x_train_pr, y_train)
        rsqu_test.append(lr.score(x_test_pr, y_test))

    plt.plot(order, rsqu_test)
    plt.xlabel('order')
    plt.ylabel('R^2')
    plt.title('R^2 Using Test Data')
    plt.text(3, 0.75, 'Maximum R^2 ')
    plt.show()


def ridge_regression():
    """
    Prevents over-fitting, which is ALSO a big problem when
    you have multiple independent variables or features

    If the estimated polynomial coefficients have a very large magnitude, we can use
    Ridge regression to control it with an 'alpha' parameter

    Alpha is a parameter we select before fitting or training a model

    As alpha increases, the other parameters get smaller
    Must be selected carefully - If alpha is too large, the parameters will reach 0, under-fitting
    the model

    if alpha is 0, over-fitting is evident!

    In order to select alpha, use cross validation
    """
    from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
    from sklearn.linear_model import Ridge, LinearRegression

    df = util.create_df()

    x_data = df.drop('price', axis=1)
    y_data = df['price']
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15,
                                                        random_state=1)

    lr = LinearRegression()
    lr.fit(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_train)

    yhat_train = lr.predict(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
    print(f'Train: {yhat_train[0:5]}')

    yhat_test = lr.predict(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
    print(f'Test: {yhat_test[0:5]}')

    title = 'Distribution Plot of Predicted Value Using Training Data vs Training Data Distribution'
    DistributionPlot(y_train, yhat_train, "Actual Values (Train)", "Predicted Values (Train)",
                     title)

    rm = Ridge(alpha=0.1)
    rm.fit(x_train, y_train)
    y_hat = rm.predict(x_test)

    print('predicted:', y_hat[0:4])
    print('test set :', y_hat[0:4].values)


def grid_search():
    """

    """
    pass


if __name__ == '__main__':
    # model_evaluation()
    fitting()
    # ridge_regression()
    # grid_search()
