from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np
import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
# from losowanie_danych import x_train, y_test, y_train, x_test

df = pd.read_csv('./dane_do_analizy.csv', sep=';')
df_val = pd.read_csv('./grupa_werfikacyjna.csv', sep=';')

x_val = df_val.iloc[:, 1:].values
y_val = df_val.iloc[:, :1].values

x = df.iloc[:, 1:].values
y = df.iloc[:, :1].values

x_train10, x_test90, y_train10, y_test90 = train_test_split(
    x, y, test_size=0.9, random_state=8
)

x_train70, x_test30, y_train70, y_test30 = train_test_split(
    x, y, test_size=0.3, random_state=8
)

wojtek_train10 = x_train10[:, [33, 62, 10, 21, 49, 15, 18, 6, 48, 51, 24, 12, 19]]
wojtek_test90 = x_test90[:, [33, 62, 10, 21, 49, 15, 18, 6, 48, 51, 24, 12, 19]]

wojtek_train70 = x_train70[:, [33, 62, 10, 21, 49, 15, 18, 6, 48, 51, 24, 12, 19]]
wojtek_test30 = x_test30[:, [33, 62, 10, 21, 49, 15, 18, 6, 48, 51, 24, 12, 19]]

x_val = x_val[:, [33, 62, 10, 21, 49, 15, 18, 6, 48, 51, 24, 12, 19]]

# Decision Tree
# dt = RandomForestClassifier(max_depth=15, random_state=0)
# dt.fit(wojtek_train10, y_train10.ravel())
# print(dt.score(wojtek_test90, y_test90))
# print(dt.score(wojtek_train10, y_train10))
# print(dt.score(x_val, y_val))


# predictions_decision_tree_1 = dt1.predict(wojtek_test90)
# fpr_dt_1, tpr_dt_1, _ = metrics.roc_curve(y_test90, predictions_decision_tree_1)
# auc_dt_1 = metrics.roc_auc_score(y_test90, predictions_decision_tree_1)

# dt2 = DecisionTreeClassifier(random_state=8)
# dt2.fit(wojtek_train70, y_train70.ravel())

# predictions_decision_tree_2 = dt2.predict(wojtek_test30)
# fpr_dt_2, tpr_dt_2, _ = metrics.roc_curve(y_test30, predictions_decision_tree_2)
# auc_dt_2 = metrics.roc_auc_score(y_test30, predictions_decision_tree_2)

# Random Forest
def get_random_forest_score(x_train, y_train, x_test, y_test, x_validation, y_validation):
    rf = RandomForestClassifier(max_depth=15, random_state=8)
    rf.fit(x_train, y_train.ravel())

    print(rf.score(x_test, y_test))
    print(rf.score(x_train, y_train))
    print(rf.score(x_validation, y_validation))


def get_decision_tree_score(x_train, y_train, x_test, y_test, x_validation, y_validation):
    dt = DecisionTreeClassifier(random_state=8)
    dt.fit(x_train, y_train.ravel())

    print(dt.score(x_test, y_test))
    print(dt.score(x_train, y_train))
    print(dt.score(x_validation, y_validation))


def get_logistic_regression_score(x_train, y_train, x_test, y_test, x_validation, y_validation):
    lg = LogisticRegression(random_state=8)
    lg.fit(x_train, y_train.ravel())

    print(lg.score(x_test, y_test))
    print(lg.score(x_train, y_train))
    print(lg.score(x_validation, y_validation))


def get_gaussian_naive_bayes_score(x_train, y_train, x_test, y_test, x_validation, y_validation):
    gnb = GaussianNB()
    gnb.fit(x_train, y_train.ravel())

    print(gnb.score(x_test, y_test))
    print(gnb.score(x_train, y_train))
    print(gnb.score(x_validation, y_validation))


# get_random_forest_score(wojtek_train70, y_train70, wojtek_test30, y_test30, x_val, y_val)
# get_decision_tree_score(wojtek_train70, y_train70, wojtek_test30, y_test30, x_val, y_val)
# get_logistic_regression_score(wojtek_train70, y_train70, wojtek_test30, y_test30, x_val, y_val)
# get_gaussian_naive_bayes_score(wojtek_train70, y_train70, wojtek_test30, y_test30, x_val, y_val)

# get_random_forest_score(wojtek_train10, y_train10, wojtek_test90, y_test90, x_val, y_val)
# get_decision_tree_score(wojtek_train10, y_train10, wojtek_test90, y_test90, x_val, y_val)
# get_logistic_regression_score(wojtek_train10, y_train10, wojtek_test90, y_test90, x_val, y_val)
# get_gaussian_naive_bayes_score(wojtek_train10, y_train10, wojtek_test90, y_test90, x_val, y_val)


def get_plot_with_two_dataset(model):


# predictions_random_forest = rf.predict(wojtek_test)
# fpr_rf, trp_fr, _ = metrics.roc_curve(y_test90, predictions_random_forest)
# auc_rf = metrics.roc_auc_score(y_test90, predictions_random_forest)
#
# # Logistic Regression
# lg = LogisticRegression(random_state=0)
# lg.fit(wojtek_train10, y_train10.ravel())
# print(lg.score(wojtek_test90, y_test90))
# print(lg.score(wojtek_train10, y_train10))
# print(lg.score(x_val, y_val))
# predictions_logistic_regression = lg.predict(wojtek_test)
# fpr_lg, trp_lg, _ = metrics.roc_curve(y_test90, predictions_logistic_regression)
# auc_lg = metrics.roc_auc_score(y_test90, predictions_logistic_regression)
#
# # Naive Bayes
# gnb = GaussianNB()
# gnb.fit(wojtek_train10, y_train10.ravel())
# print(gnb.score(wojtek_test90, y_test90))
# print(gnb.score(wojtek_train10, y_train10))
# print(gnb.score(x_val, y_val))
# predictions_gaussian_naive_bayes = gnb.predict(wojtek_test)
# fpr_gnb, trp_gnb, _ = metrics.roc_curve(y_test90, predictions_gaussian_naive_bayes)
# auc_gnb = metrics.roc_auc_score(y_test90, predictions_gaussian_naive_bayes)

# print('decision tree')
# print(confusion_matrix(y_test90, predictions_decision_tree_1))
# print(confusion_matrix(y_test30, predictions_decision_tree_2))

# print('random forest')
# print(confusion_matrix(y_test90, predictions_random_forest))
# print('logistic regression')
# print(confusion_matrix(y_test90, predictions_logistic_regression))
# print('gaussian naive bayes')
# print(confusion_matrix(y_test90, predictions_gaussian_naive_bayes))


# plt.plot(fpr_dt_1, tpr_dt_1, label="Test=90% AUC=" + str(auc_dt_1))
# plt.plot(fpr_dt_2, tpr_dt_2, label="Test=30% AUC=" + str(auc_dt_2))

# plt.plot(fpr_rf, trp_fr, label="Random Forest AUC="+str(auc_rf))
# plt.plot(fpr_lg, trp_lg, label="Logistic Regression AUC="+str(auc_lg))
# plt.plot(fpr_gnb, trp_gnb, label="Gaussian Naive Bayes AUC="+str(auc_gnb))
# plt.title("ROC Curve Decision Tree")
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.legend(loc=4)
# plt.show()
