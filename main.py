import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from methods import *
from sklearn.ensemble import VotingClassifier


def make_hybrid_x(methods):
    hybrid_train = np.array([methods[0]])
    for i in range(1, methods.__len__()):
        hybrid_train = np.r_[hybrid_train, [methods[i]]]

    hybrid_train = hybrid_train.transpose()
    return hybrid_train


def make_hybrid_model(x_train, x_test, y_train, y_test):
    wojtek_train = x_train[:, [33, 62, 10, 21, 49, 15, 18, 6, 48, 51, 24, 12, 19]]
    wojtek_test = x_test[:, [33, 62, 10, 21, 49, 15, 18, 6, 48, 51, 24, 12, 19]]
    # wojtek_val = x_val[:, [33, 62, 10, 21, 49, 15, 18, 6, 48, 51, 24, 12, 19]]

    jakub_train = x_train[:, [7, 8, 22, 26, 29, 34, 35, 37, 40, 45, 47, 52, 61]]
    jakub_test = x_test[:, [7, 8, 22, 26, 29, 34, 35, 37, 40, 45, 47, 52, 61]]
    # jakub_val = x_val[:, [7, 8, 22, 26, 29, 34, 35, 37, 40, 45, 47, 52, 61]]

    szymon_train = x_train[:, [1, 9, 16, 20, 36, 38, 41, 42, 43, 44, 53, 58, 59]]
    szymon_test = x_test[:, [1, 9, 16, 20, 36, 38, 41, 42, 43, 44, 53, 58, 59]]
    # szymon_val = x_val[:, [1, 9, 16, 20, 36, 38, 41, 42, 43, 44, 53, 58, 59]]

    maciek_train = x_train[:, [9, 50, 54, 5, 32, 27, 28, 30, 25, 11, 3, 60, 0]]
    maciek_test = x_test[:, [9, 50, 54, 5, 32, 27, 28, 30, 25, 11, 3, 60, 0]]
    # maciek_val = x_val[:, [9, 50, 54, 5, 32, 27, 28, 30, 25, 11, 3, 60, 0]]

    bartek_train = x_train[:, [23, 55, 57, 17, 56, 13, 39, 46, 31, 14, 63, 64, 2]]
    bartek_test = x_test[:, [23, 55, 57, 17, 56, 13, 39, 46, 31, 14, 63, 64, 2]]
    # bartek_val = x_val[:, [23, 55, 57, 17, 56, 13, 39, 46, 31, 14, 63, 64, 2]]

    # wojtek: decision tree, random forest, naive bayes, logistic regression
    # jakub: knn, random forest, svc, naive bayes
    # szymon: linear_discriminant, natural_network, gradient_boosting, radius_neighbors
    # maciek: logistic regression, sgd, natural_network, gradient_boosting
    # bartek: knn, svc, decision tree, svr

    wojtek_dt = decision_tree(wojtek_train, y_train, wojtek_test)
    wojtek_rf = random_forest(wojtek_train, y_train, wojtek_test)
    wojtek_gnb = gaussian_naive_bayes(wojtek_train, y_train, wojtek_test)
    wojtek_lg = logistic_regression(wojtek_train, y_train, wojtek_test)

    jakub_rf = random_forest(jakub_train, y_train, jakub_test)
    jakub_gnb = gaussian_naive_bayes(jakub_train, y_train, jakub_test)
    jakub_knn = KNN(jakub_train, y_train, jakub_test)
    jakub_svc = SVC(jakub_train, y_train, jakub_test)

    bartek_dt = decision_tree(bartek_train, y_train, bartek_test)
    bartek_knn = KNN(bartek_train, y_train, bartek_test)
    bartek_svc = SVC(bartek_train, y_train, bartek_test)
    # bartek_svr = svr(bartek_train, y_train, bartek_test)

    maciek_lg = logistic_regression(maciek_train, y_train, maciek_test)
    maciek_nn = natural_network(maciek_train, y_train, maciek_test)
    maciek_gb = gradient_boosting(maciek_train, y_train, maciek_test)
    maciek_sgd = SGD(maciek_train, y_train, maciek_test)

    szymon_ld = linear_discriminant_analysis(szymon_train, y_train, szymon_test)
    szymon_nn = natural_network(szymon_train, y_train, szymon_test)
    szymon_gb = gradient_boosting(szymon_train, y_train, szymon_test)
    szymon_rn = radius_neighbors(szymon_train, y_train, szymon_test)

    models =  [
        wojtek_dt,
        wojtek_rf,
        wojtek_gnb,
        wojtek_lg,
        jakub_rf,
        jakub_gnb,
        jakub_knn,
        jakub_svc,
        bartek_dt,
        bartek_knn,
        bartek_svc,
        # bartek_svr,
        maciek_lg,
        maciek_nn,
        maciek_gb,
        maciek_sgd,
        szymon_ld,
        szymon_nn,
        szymon_gb,
        szymon_rn,
    ]
    for model in  models:
        print(score(model,y_test))
    return models

df = pd.read_csv('./Arkusz2.csv', sep=';')

x = df.iloc[:, 1:].values
y = df.iloc[:, :1].values

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=8
)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=8)

methods = make_hybrid_model(x_train, x_train, y_train, y_train)
hybrid_train = make_hybrid_x(methods)
hybrid_model = natural_network(hybrid_train, y_train, hybrid_train)

print(hybrid_model)
print(score(hybrid_model, y_train))
print('!!!!')
methods2 = make_hybrid_model(x_train, x_test, y_train, y_test)
hybrid_test = make_hybrid_x(methods2)
hybrid_model2 = natural_network(hybrid_train, y_train, hybrid_test)

print(hybrid_model2)
print(score(hybrid_model2, y_test))
print('!!!!')

# eclf1 = eclf1.fit(x, y)
# print(eclf1.predict(x))
