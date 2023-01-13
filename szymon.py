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
from losowanie_danych import x_train, x_test, y_train, y_test, x_val, y_val
from methods import Wykres


def parse_data():
    data = pd.read_excel(r'szymon/Z02A.xlsx', 'Szymon', skiprows=4)
    data_frame = pd.DataFrame(data)
    return data_frame


def all_data(data):
    return chose_range(data, 0, 13)


def data70_30(data):
    study = chose_range(data, 17, 30)
    study = remove_empty(study)
    test = chose_range(data, 34, 47)
    test = remove_empty(test)

    return study, test


def data10_90(data):
    study = chose_range(data, 51, 64)
    study = remove_empty(study)
    test = chose_range(data, 68, 81)
    test = remove_empty(test)

    return study, test


def chose_range(data, a, b):
    chosed = data.drop(data.iloc[:, 0: a], axis=1)
    chosed = chosed.drop(data.iloc[:, (b + 1): 84], axis=1)
    return chosed


def remove_empty(data):
    for row in range(0, len(data.index)):
        if data.iloc[row][0] != data.iloc[row][0]:
            rows = np.arange((row), len(data.index))
            data = data.drop(rows, axis=0)
            return data


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


def analize(study, test):
    lda = linear_discriminant_analysis(study)
    nn = natural_network(study)
    gb = gradient_boosting(study)
    rn = radius_neighbors(study)

    plot(lda, test, 'Analiza dyskrymiancyja')
    plot(nn, test, 'Sieć Neuronowa')
    plot(gb, test, 'Gradient Boosting')
    plot(rn, test, 'Radius Neighbors')
    plt.show()
    plt.clf()
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


train, test, val = (make_data(x_train, x_test, x_val, y_train, y_test, y_val))
#print('train', 'train')

#print(analize(train, train))

print('train', 'test')
plt.title('Testowa')
print(analize(train, test))

print('train', 'val')
plt.title('Walidacyjna')
print(analize(train, val))


