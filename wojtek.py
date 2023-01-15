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

def get_random_forest_score(x_train, y_train, x_test, y_test, x_validation, y_validation):
    rf = RandomForestClassifier(max_depth=15, random_state=8)
    rf.fit(x_train, y_train.ravel())
    predictions_random_forest = rf.predict(x_test)

    print(rf.score(x_test, y_test))
    print(rf.score(x_train, y_train))
    print(rf.score(x_validation, y_validation))
    print('random forest')
    print(confusion_matrix(y_test, predictions_random_forest))


def get_decision_tree_score(x_train, y_train, x_test, y_test, x_validation, y_validation):
    dt = DecisionTreeClassifier(random_state=8)
    dt.fit(x_train, y_train.ravel())
    predictions_decision_tree = dt.predict(x_test)

    print(dt.score(x_test, y_test))
    print(dt.score(x_train, y_train))
    print(dt.score(x_validation, y_validation))
    print('decision tree')
    print(confusion_matrix(y_test, predictions_decision_tree))


def get_logistic_regression_score(x_train, y_train, x_test, y_test, x_validation, y_validation):
    lg = LogisticRegression(random_state=8)
    lg.fit(x_train, y_train.ravel())
    predictions_logistic_regression = lg.predict(x_test)
    print(lg.score(x_test, y_test))
    print(lg.score(x_train, y_train))
    print(lg.score(x_validation, y_validation))
    print('logistic regression')
    print(confusion_matrix(y_test, predictions_logistic_regression))


def get_gaussian_naive_bayes_score(x_train, y_train, x_test, y_test, x_validation, y_validation):
    gnb = GaussianNB()
    gnb.fit(x_train, y_train.ravel())
    predictions_gaussian_naive_bayes = gnb.predict(x_test)
    print(gnb.score(x_test, y_test))
    print(gnb.score(x_train, y_train))
    print(gnb.score(x_validation, y_validation))
    print('gaussian naive bayes')
    print(confusion_matrix(y_test, predictions_gaussian_naive_bayes))


# get_random_forest_score(wojtek_train70, y_train70, wojtek_test30, y_test30, x_val, y_val)
# get_decision_tree_score(wojtek_train70, y_train70, wojtek_test30, y_test30, x_val, y_val)
get_logistic_regression_score(wojtek_train70, y_train70, wojtek_test30, y_test30, x_val, y_val)
# get_gaussian_naive_bayes_score(wojtek_train70, y_train70, wojtek_test30, y_test30, x_val, y_val)

# get_random_forest_score(wojtek_train10, y_train10, wojtek_test90, y_test90, x_val, y_val)
# get_decision_tree_score(wojtek_train10, y_train10, wojtek_test90, y_test90, x_val, y_val)
# get_logistic_regression_score(wojtek_train10, y_train10, wojtek_test90, y_test90, x_val, y_val)
# get_gaussian_naive_bayes_score(wojtek_train10, y_train10, wojtek_test90, y_test90, x_val, y_val)


# predictions_random_forest = rf.predict(wojtek_test)
# fpr_rf, trp_fr, _ = metrics.roc_curve(y_test90, predictions_random_forest)
# auc_rf = metrics.roc_auc_score(y_test90, predictions_random_forest)


def get_plot_for_all_methods(x_train, y_train, x_test, y_test, title):
    # Decision Tree
    dt = DecisionTreeClassifier(random_state=0, max_depth=5)
    dt.fit(x_train, y_train.ravel())
    predictions_decision_tree = dt.predict_proba(x_test)[::, 1]
    fpr_dt, trp_dt, _ = metrics.roc_curve(y_test.ravel(), predictions_decision_tree)
    auc_dt = metrics.roc_auc_score(y_test, predictions_decision_tree)

    # Random Forest
    rf = RandomForestClassifier(max_depth=15, random_state=8)
    rf.fit(x_train, y_train.ravel())
    predictions_random_forest = rf.predict_proba(x_test)[::, 1]
    fpr_rf, trp_rf, _ = metrics.roc_curve(y_test.ravel(), predictions_random_forest)
    auc_rf = metrics.roc_auc_score(y_test, predictions_random_forest)

    # Logistic Regression
    lg = LogisticRegression(random_state=8)
    lg.fit(x_train, y_train.ravel())
    predictions_logistic_regression = lg.predict_proba(x_test)[::, 1]
    fpr_lg, trp_lg, _ = metrics.roc_curve(y_test.ravel(), predictions_logistic_regression)
    auc_lg = metrics.roc_auc_score(y_test, predictions_logistic_regression)

    # Naive Bayes
    gnb = GaussianNB()
    gnb.fit(x_train, y_train.ravel())
    predictions_gaussian_naive_bayes = gnb.predict_proba(x_test)[::, 1]
    fpr_gnb, trp_gnb, _ = metrics.roc_curve(y_test, predictions_gaussian_naive_bayes)
    auc_gnb = metrics.roc_auc_score(y_test, predictions_gaussian_naive_bayes)

    plt.plot(fpr_dt, trp_dt, label="Decision Tree AUC="+str(auc_dt))
    plt.plot(fpr_rf, trp_rf, label="Random Forest AUC="+str(auc_rf))
    plt.plot(fpr_lg, trp_lg, label="Logistic Regression AUC="+str(auc_lg))
    plt.plot(fpr_gnb, trp_gnb, label="Gaussian Naive Bayes AUC="+str(auc_gnb))
    plt.title(title)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.show()


def get_plots_for_decision_tree():
    dt1 = DecisionTreeClassifier(random_state=8, max_depth=5)
    dt1.fit(wojtek_train10, y_train10.ravel())
    predictions_decision_tree1 = dt1.predict_proba(wojtek_test90)[::, 1]
    fpr_dt1, trp_dt1, _ = metrics.roc_curve(y_test90.ravel(), predictions_decision_tree1)
    auc_dt1 = metrics.roc_auc_score(y_test90, predictions_decision_tree1)

    dt2 = DecisionTreeClassifier(random_state=8, max_depth=5)
    dt2.fit(wojtek_train70, y_train70.ravel())
    predictions_decision_tree2 = dt1.predict_proba(wojtek_test30)[::, 1]
    fpr_dt2, trp_dt2, _ = metrics.roc_curve(y_test30.ravel(), predictions_decision_tree2)
    auc_dt2 = metrics.roc_auc_score(y_test30, predictions_decision_tree2)

    plt.plot(fpr_dt1, trp_dt1, label="Train 10% AUC=" + str(auc_dt1))
    plt.plot(fpr_dt2, trp_dt2, label="Train 70% AUC=" + str(auc_dt2))
    plt.title("Curve ROC Decision tree")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.show()


def get_plots_for_random_forest():
    dt1 = RandomForestClassifier(max_depth=15, random_state=8)
    dt1.fit(wojtek_train10, y_train10.ravel())
    predictions_decision_tree1 = dt1.predict_proba(wojtek_test90)[::, 1]
    fpr_dt1, trp_dt1, _ = metrics.roc_curve(y_test90.ravel(), predictions_decision_tree1)
    auc_dt1 = metrics.roc_auc_score(y_test90, predictions_decision_tree1)

    dt2 = RandomForestClassifier(max_depth=15, random_state=8)
    dt2.fit(wojtek_train70, y_train70.ravel())
    predictions_decision_tree2 = dt1.predict_proba(wojtek_test30)[::, 1]
    fpr_dt2, trp_dt2, _ = metrics.roc_curve(y_test30.ravel(), predictions_decision_tree2)
    auc_dt2 = metrics.roc_auc_score(y_test30, predictions_decision_tree2)

    plt.plot(fpr_dt1, trp_dt1, label="Train 10% AUC=" + str(auc_dt1))
    plt.plot(fpr_dt2, trp_dt2, label="Train 70% AUC=" + str(auc_dt2))
    plt.title("Curve ROC Random Forest")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.show()


def get_plots_for_logistic_regression():
    dt1 = LogisticRegression(random_state=8)
    dt1.fit(wojtek_train10, y_train10.ravel())
    predictions_decision_tree1 = dt1.predict_proba(wojtek_test90)[::, 1]
    fpr_dt1, trp_dt1, _ = metrics.roc_curve(y_test90.ravel(), predictions_decision_tree1)
    auc_dt1 = metrics.roc_auc_score(y_test90, predictions_decision_tree1)

    dt2 = LogisticRegression(random_state=8)
    dt2.fit(wojtek_train70, y_train70.ravel())
    predictions_decision_tree2 = dt1.predict_proba(wojtek_test30)[::, 1]
    fpr_dt2, trp_dt2, _ = metrics.roc_curve(y_test30.ravel(), predictions_decision_tree2)
    auc_dt2 = metrics.roc_auc_score(y_test30, predictions_decision_tree2)

    plt.plot(fpr_dt1, trp_dt1, label="Train 10% AUC=" + str(auc_dt1))
    plt.plot(fpr_dt2, trp_dt2, label="Train 70% AUC=" + str(auc_dt2))
    plt.title("Curve ROC Logistic Regression")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.show()


def get_plots_for_gaussian_naive_bayes():
    dt1 = GaussianNB()
    dt1.fit(wojtek_train10, y_train10.ravel())
    predictions_decision_tree1 = dt1.predict_proba(wojtek_test90)[::, 1]
    fpr_dt1, trp_dt1, _ = metrics.roc_curve(y_test90.ravel(), predictions_decision_tree1)
    auc_dt1 = metrics.roc_auc_score(y_test90, predictions_decision_tree1)

    dt2 = GaussianNB()
    dt2.fit(wojtek_train70, y_train70.ravel())
    predictions_decision_tree2 = dt1.predict_proba(wojtek_test30)[::, 1]
    fpr_dt2, trp_dt2, _ = metrics.roc_curve(y_test30.ravel(), predictions_decision_tree2)
    auc_dt2 = metrics.roc_auc_score(y_test30, predictions_decision_tree2)

    plt.plot(fpr_dt1, trp_dt1, label="Train 10% AUC=" + str(auc_dt1))
    plt.plot(fpr_dt2, trp_dt2, label="Train 70% AUC=" + str(auc_dt2))
    plt.title("Curve ROC Gaussian Naive Bayes")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.show()

# get_plot_for_all_methods(wojtek_train10, y_train10, wojtek_test90, y_test90, "ROC Curve train 10%")
# get_plot_for_all_methods(wojtek_train70, y_train70, wojtek_test30, y_test30, "ROC Curve train 70%")

# get_plots_for_decision_tree()
# get_plots_for_random_forest()
# get_plots_for_logistic_regression()
get_plots_for_gaussian_naive_bayes()
