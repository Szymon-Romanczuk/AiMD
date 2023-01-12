import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from methods import *
from sklearn.ensemble import VotingClassifier

df = pd.read_csv('./dane_do_analizy.csv', sep=';')
df_val = pd.read_csv('./grupa_werfikacyjna.csv', sep=';')

x = df.iloc[:, 1:].values
y = df.iloc[:, :1].values

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=8
)

x_val = df_val.iloc[:, 1:].values
y_val = df_val.iloc[:, :1].values


wojtek_train = x_train[:, [33, 62, 10, 21, 49, 15, 18, 6, 48, 51, 24, 12, 19]]
wojtek_test = x_test[:, [33, 62, 10, 21, 49, 15, 18, 6, 48, 51, 24, 12, 19]]
wojtek_val = x_val[:, [33, 62, 10, 21, 49, 15, 18, 6, 48, 51, 24, 12, 19]]

jakub_train = x_train[:, [7, 8, 22, 26, 29, 34, 35, 37, 40, 45, 47, 52, 61]]
jakub_test = x_test[:, [7, 8, 22, 26, 29, 34, 35, 37, 40, 45, 47, 52, 61]]
jakub_val = x_val[:, [7, 8, 22, 26, 29, 34, 35, 37, 40, 45, 47, 52, 61]]

szymon_train = x_train[:, [1, 9, 16, 20, 36, 38, 41, 42, 43, 44, 53, 58, 59]]
szymon_test = x_test[:, [1, 9, 16, 20, 36, 38, 41, 42, 43, 44, 53, 58, 59]]
szymon_val = x_val[:, [1, 9, 16, 20, 36, 38, 41, 42, 43, 44, 53, 58, 59]]

maciek_train = x_train[:, [9, 50, 54, 5, 32, 27, 28, 30, 25, 11, 3, 60, 0]]
maciek_test = x_test[:, [9, 50, 54, 5, 32, 27, 28, 30, 25, 11, 3, 60, 0]]
maciek_val = x_val[:, [9, 50, 54, 5, 32, 27, 28, 30, 25, 11, 3, 60, 0]]

bartek_train = x_train[:, [23, 55, 57, 17, 56, 13, 39, 46, 31, 14, 63, 64, 2]]
bartek_test = x_test[:, [23, 55, 57, 17, 56, 13, 39, 46, 31, 14, 63, 64, 2]]
bartek_val = x_val[:, [23, 55, 57, 17, 56, 13, 39, 46, 31, 14, 63, 64, 2]]

#wojtek: decision tree, random forest, naive bayes, logistic regression
#jakub: knn, random forest, svc, naive bayes
#szymon: linear_discriminant, natural_network, gradient_boosting, radius_neighbors
#maciek: logistic regression, sgd, natural_network, gradient_boosting
#bartek: knn, svc, decision tree, svr

wojtek_dt = decision_tree(wojtek_train, y_train)
wojtek_rf = random_forest(wojtek_train, y_train)
wojtek_gnb = gaussian_naive_bayes(wojtek_train, y_train)
wojtek_lg = logistic_regression(wojtek_train, y_train)

jakub_rf = random_forest(jakub_train, y_train)
jakub_gnb = gaussian_naive_bayes(jakub_train, y_train)
jakub_knn = KNN(jakub_train, y_train)
jakub_svc = SVC(jakub_train, y_train)

bartek_dt = decision_tree(bartek_train, y_train)
bartek_knn = KNN(bartek_train, y_train)
bartek_svc = SVC(bartek_train, y_train)
bartek_svr = svr(bartek_train, y_train)

maciek_lg = logistic_regression(maciek_train, y_train)
maciek_nn = natural_network(maciek_train, y_train)
maciek_gb = gradient_boosting(maciek_train, y_train)
maciek_sgd = SGD(maciek_train, y_train)

szymon_ld = linear_discriminant_analysis(szymon_train, y_train)
szymon_nn = natural_network(szymon_train, y_train)
szymon_gb = gradient_boosting(szymon_train, y_train)
szymon_rn = radius_neighbors(szymon_train, y_train)

eclf1 = VotingClassifier(estimators=[
    ('wdt', wojtek_dt),
    ('wrf', wojtek_rf),
    ('wgnb', wojtek_gnb),
    ('wlg', wojtek_lg),
    ('jrf', jakub_rf),
    ('jgnb', jakub_gnb),
    ('jknn', jakub_knn),
    ('jsvr', jakub_svc),
    ('bdt', bartek_dt),
    ('bknn', bartek_knn),
    ('bsvc', bartek_svc),
    ('bsvr', bartek_svr),
    ('mlg', maciek_lg),
    ('mnn', maciek_nn),
    ('mgb', maciek_gb),
    ('msgd', maciek_sgd),
    ('sld', szymon_ld),
    ('snn', szymon_nn),
    ('sgb', szymon_gb),
    ('sr', szymon_rn),
], voting='soft')
# eclf1 = eclf1.fit(x, y)
# print(eclf1.predict(x))