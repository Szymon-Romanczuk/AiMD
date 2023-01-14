import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from losowanie_danych import x_train, y_train, x_test, y_test


# fit the model on the training data with all possible predictors
def stepwise_linear_regression(x_training, y_training, x_testing, y_testing):
    X_train = sm.add_constant(x_training)
    model = sm.OLS(y_training, x_training).fit()

    # get the p-values for all predictors
    p_values = model.pvalues

    # initialize an empty list to store the selected predictors
    predictors = []

    # iterate over the p-values and add the predictor to the list if its p-value is less than the threshold
    threshold = 0.05
    for i in range(len(p_values)):
        if p_values[i] < threshold:
            predictors.append(i)

    # fit the model again with the selected predictors
    X_train_new = X_train[:, predictors]
    X_train_new = sm.add_constant(X_train_new)
    model_new = sm.OLS(y_train, X_train_new).fit()

    X_test = sm.add_constant(x_test)
    X_test_new = X_test[:, predictors]
    X_test_new = sm.add_constant(X_test_new)

    # get the score on the test data

    y_pred = model_new.predict(X_test_new)
    # print(len(y_pred))
    score = r2_score(y_testing, y_pred)
    print(score)


# Backwards
def backwards_linear_regression(x_training, y_training, x_testing, y_testing):
    # fit the model with all possible predictors
    X_train = sm.add_constant(x_training)
    model = sm.OLS(y_training, x_training).fit()

    # get the p-values for all predictors
    p_values = model.pvalues

    # initialize a list to store the selected predictors
    predictors = list(range(len(p_values)))

    # set the threshold for significance level
    threshold = 0.05

    p_values = pd.DataFrame(p_values)
    # iterate over the p-values and remove the predictor if its p-value is greater than the threshold
    while max(p_values[0]) > threshold:
        idx_max = p_values[0].idxmax()
        predictors = [p for i, p in enumerate(predictors) if i != idx_max]
        p_values = p_values.drop(idx_max)

    # fit the model again with the selected predictors
    X_train_new = X_train[:, predictors]
    X_train_new = sm.add_constant(X_train_new)
    model_new = sm.OLS(y_train, X_train_new).fit()

    X_test = sm.add_constant(x_test)
    X_test_new = X_test[:, predictors]
    X_test_new = sm.add_constant(X_test_new)

    # get the score on the test data

    y_pred = model_new.predict(X_test_new)
    # print(len(y_pred))
    score = r2_score(y_testing, y_pred)
    print(score)


backwards_linear_regression(x_train, y_train, x_test, y_test)
stepwise_linear_regression(x_train, y_train, x_test, y_test)
