# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from losowanie_danych import r10_90, r70_30
from methods import Wykres


def linear_discriminant_analysis(study):
    X_study, Y_study = sapareteXY(study)

    clf = LinearDiscriminantAnalysis()
    clf.fit(X_study, Y_study)
    return clf


def natural_network(study):
    X_study, Y_study = sapareteXY(study)

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(9, 5, 3), random_state=1, max_iter=6000)
    # 9,5,3
    clf.fit(X_study, Y_study)
    return clf


def gradient_boosting(study):
    X_study, Y_study = sapareteXY(study)
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    clf.fit(X_study, Y_study)

    return clf


def radius_neighbors(study):
    X_study, Y_study = sapareteXY(study)
    clf = RadiusNeighborsClassifier(radius=1.8)
    clf.fit(X_study, Y_study)
    return clf


def test_model(clf, test):
    X, Y = sapareteXY(test)
    return clf.score(X, Y)


def plot(clf, test, title):
    X, Y = sapareteXY(test)
    y_predict = clf.predict(X)
    Wykres(y_predict, Y, title)


def sapareteXY(data):
    np_data = data
    Y = np_data[:, 0]
    X = np.delete(np_data, 0, 1)
    return X, Y


def analize(study, test, name):
    lda = linear_discriminant_analysis(study)
    nn = natural_network(study)
    gb = gradient_boosting(study)
    rn = radius_neighbors(study)

    #plot(lda, test, name)
    #plot(nn, test, name)
    #plot(gb, test, name)
    plot(rn, test, name)
    #plt.show()
    #plt.clf()
    return {
        'Analiza dyskryminacyjna': test_model(lda, test),
        'Sieć Neuronowa': test_model(nn, test),
        'Gradient Boosting': test_model(gb, test),
        'Radius Neighbors': test_model(rn, test),
    }


def make_data(x_train, x_test, x_val, y_train, y_test, y_val):
    szymon_train = x_train[:, [1, 9, 16, 20, 36, 38, 41, 42, 43, 44, 53, 58, 59]]
    szymon_test = x_test[:, [1, 9, 16, 20, 36, 38, 41, 42, 43, 44, 53, 58, 59]]
    szymon_val = x_val[:, [1, 9, 16, 20, 36, 38, 41, 42, 43, 44, 53, 58, 59]]
    szymon_train = np.c_[y_train, szymon_train]
    szymon_test = np.c_[y_test, szymon_test]
    szymon_val = np.c_[y_val, szymon_val]
    return szymon_train, szymon_test, szymon_val


x_train, x_test, x_val, y_train, y_test, y_val = r10_90()
train, test, val = (make_data(x_train, x_test, x_val, y_train, y_test, y_val))

print('train', 'test')
plt.title('Radius Neighbors')
print(analize(train, test, '10 90 AUC = '))

x_train, x_test, x_val, y_train, y_test, y_val = r70_30()
train, test, val = (make_data(x_train, x_test, x_val, y_train, y_test, y_val))

print(analize(train, test, '70 30 AUC = '))
plt.show()
