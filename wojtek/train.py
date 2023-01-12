from sklearn.model_selection import train_test_split
import numpy as np
import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

import pandas as pd


def chose_range(data, a, b):
    chose = data.drop(data.iloc[:, 0: a], axis=1)
    chose = chose.drop(data.iloc[:, (b+1): 84], axis=1)
    return chose


def remove_empty(data):
    for row in range(0, len(data.index)):
        if data.iloc[row][0] != data.iloc[row][0]:
            rows = np.arange(row, len(data.index))
            data = data.drop(rows, axis=0)
            return data


def separate_x_and_y(data):
    x = data.iloc[:,1:].values
    y = data.iloc[:,:1].values
    return y, x


# Importing dataset
# df = pd.read_excel("./data.xlsx")
# print(df)
# x = df.iloc[:, 1:].values
# y = df.iloc[:, :1].values




def decision_tree(x_train, y_train, x_test, y_test):
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(x_train, y_train)
    predictionsDecisionTree = clf.predict(x_test)
    score_decision_tree = clf.score(x_test, y_test)

    return score_decision_tree


def random_forest(x_train, y_train, x_test, y_test):
    rf = RandomForestClassifier(max_depth=15, random_state=0)
    rf.fit(x_train, y_train.ravel())
    predictionsRandomForest = rf.predict(x_test)
    score_random_forest = rf.score(x_test, y_test)

    return score_random_forest


def gaussian_naive_bayes(x_train, y_train, x_test, y_test):
    gnb = GaussianNB()
    gnb.fit(x_train, y_train.ravel())
    predictionsNaiveBayes = gnb.predict(x_test)
    score_naive_bayes = gnb.score(x_test, y_test)
    return  score_naive_bayes


def logistic_regression(x_train, y_train, x_test, y_test):
    lg = LogisticRegression(random_state=0)
    lg.fit(x_train, y_train.ravel())
    predictionsLogisticRegression = lg.predict(x_test)
    score_logistic_regression = lg.score(x_train, y_train)

    return score_logistic_regression


def get_scores():
    df = pd.read_excel('../wojtek_dane.xlsx')

    whole_data = chose_range(df, 0, 13)
    # print(whole_dat)

    train70 = chose_range(df, 16, 29)
    train70 = remove_empty(train70)

    test30 = chose_range(df, 32, 45)
    test30 = remove_empty(test30)

    train10 = chose_range(df, 48, 61)
    train10 = remove_empty(train10)

    test90 = chose_range(df, 64, 77)
    test90 = remove_empty(test90)

    y_train100, x_train100 = separate_x_and_y(whole_data)
    y_test0, x_test0 = separate_x_and_y(whole_data)

    y_train70, x_train70 = separate_x_and_y(train70)
    y_test30, x_test30 = separate_x_and_y(test30)

    y_test90, x_test90 = separate_x_and_y(test90)
    y_train10, x_train10 = separate_x_and_y(train10)

    dt70_30 = decision_tree(x_train70, y_train70, x_test30, y_test30)
    dt10_90 = decision_tree(x_train10, y_train10, x_test90, y_test90)
    dt100_0 = decision_tree(x_train100, y_train100, x_test0, y_test0)

    rf70_30 = random_forest(x_train70, y_train70, x_test30, y_test30)
    rf10_90 = random_forest(x_train10, y_train10, x_test90, y_test90)
    rf100_0 = random_forest(x_train100, y_train100, x_test0, y_test0)

    gnb70_30 = gaussian_naive_bayes(x_train70, y_train70, x_test30, y_test30)
    gnb10_90 = gaussian_naive_bayes(x_train10, y_train10, x_test90, y_test90)
    gnb100_0 = gaussian_naive_bayes(x_train100, y_train100, x_test0, y_test0)

    lr70_30 = logistic_regression(x_train70, y_train70, x_test30, y_test30)
    lr10_90 = logistic_regression(x_train10, y_train10, x_test90, y_test90)
    lr100_0 = logistic_regression(x_train100, y_train100, x_test0, y_test0)

    return {
      "70_30": {
          'decision_tree': dt70_30,
          'random_forest': rf70_30,
          'naive_bayes': gnb70_30,
          'logistic_regression': lr70_30,
      },
      "10_90": {
          'decision_tree': dt10_90,
          'random_forest': rf10_90,
          'naive_bayes': gnb10_90,
          'logistic_regression': lr10_90,
      },
      "100_0": {
          'decision_tree': dt100_0,
          'random_forest': rf100_0,
          'naive_bayes': gnb100_0,
          'logistic_regression': lr100_0,
      }
    }


scores = get_scores()
print(scores)
