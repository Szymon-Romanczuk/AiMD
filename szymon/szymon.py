# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier
from sklearn.neighbors import RadiusNeighborsClassifier


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


def sapareteXY(data):
    np_data = data.to_numpy()
    Y = np_data[:, 0]
    X = np.delete(np_data, 0, 1)
    return X, Y


def analize(study, test):
    lda = linear_discriminant_analysis(study)
    nn = natural_network(study)
    gb = gradient_boosting(study)
    rn = radius_neighbors(study)

    return {
        'Analiza dyskryminacyjna': test_model(lda, test),
        'Sieć Neuronowa': test_model(nn, test),
        'Gradient Boosting': test_model(gb, test),
        'Radius Neighbors': test_model(rn, test),
    }


def SR():
    data = parse_data()
    study70, test30 = data70_30(data)
    study10, test90 = data10_90(data)
    #print(analize(study70, test30))

    return {
        "70_30": analize(study70, test30),
        "10_90": analize(study10, test90),
        "70_30_on_train": analize(study70, study70),
        "10_90_on_train": analize(study10, study10),
    }


print(SR())
