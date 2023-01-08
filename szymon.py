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
    data = pd.read_excel(r'Z02A.xlsx', 'Szymon', skiprows=4)
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
    chosed = chosed.drop(data.iloc[:, (b+1): 84], axis=1)
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
    #9,5,3
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


def hybrid_model(study):
    X_study, Y_study = sapareteXY(study)
    #clf1 = linear_discriminant_analysis(study)
    clf2 = natural_network(study)
    clf3 = gradient_boosting(study)
    #clf4 = radius_neighbors(study)

    eclf1 = VotingClassifier(estimators=[
        #('lda', clf1),
        ('nn', clf2),
        ('gb', clf3),
        #('rn', clf4),
        ], voting='soft')
    eclf1 = eclf1.fit(X_study, Y_study)
    return eclf1


def analize(study, test):
    clf = linear_discriminant_analysis(study)
    print("Analiza dyskryminacjyjna:", test_model(clf, test))
    clf = natural_network(study)
    print("Sieć Neuronowa:", test_model(clf, test))
    clf = gradient_boosting(study)
    print("Gradient Boosting:", test_model(clf, test))
    clf = radius_neighbors(study)
    print("Radius Neighbors:", test_model(clf, test))
    clf = hybrid_model(study)
    print("Hybrid Model:", test_model(clf, test))
    print()


data = parse_data()

print("Całość")
all_data = all_data(data)
analize(all_data, all_data)

print("70/30")
study, test = data70_30(data)
analize(study, test)

print("10/90")
study, test = data10_90(data)
analize(study, test)
