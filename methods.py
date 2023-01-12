from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier


def decision_tree(x_train, y_train, x_test):
    dt = DecisionTreeClassifier(random_state=0)
    dt.fit(x_train, y_train.ravel())

    return dt.predict(x_test)


def random_forest(x_train, y_train, x_test):
    rf = RandomForestClassifier(max_depth=15, random_state=0)
    rf.fit(x_train, y_train.ravel())

    return rf.predict(x_test)


def gaussian_naive_bayes(x_train, y_train, x_test):
    gnb = GaussianNB()
    gnb.fit(x_train, y_train.ravel())

    return gnb.predict(x_test)


def logistic_regression(x_train, y_train, x_test):
    lg = LogisticRegression(random_state=0)
    lg.fit(x_train, y_train.ravel())

    return lg.predict(x_test)


def KNN(x_train, y_train, x_test):
    knn = KNeighborsClassifier(n_neighbors=2)  # n_neighbors=10
    knn.fit(x_train, y_train.ravel())
    return knn.predict(x_test)


def SVC(x_train, y_train, x_test):
    from sklearn.svm import SVC
    svc = SVC(probability=True, kernel='linear')
    svc.fit(x_train, y_train.ravel())

    return svc.predict(x_test)


def linear_discriminant_analysis(x_train, y_train, x_test):
    clf = LinearDiscriminantAnalysis()
    clf.fit(x_train, y_train.ravel())

    return clf.predict(x_test)


def natural_network(x_train, y_train, x_test):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(14, 8, 6, 3, 2), random_state=1, max_iter=6000)
    clf.fit(x_train, y_train.ravel())

    return clf.predict(x_test)


def gradient_boosting(x_train, y_train, x_test):
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    clf.fit(x_train, y_train.ravel())

    return clf.predict(x_test)


def radius_neighbors(x_train, y_train, x_test):
    clf = RadiusNeighborsClassifier(radius=1.8)
    clf.fit(x_train, y_train.ravel())

    return clf.predict(x_test)


def svr(x_train, y_train, x_test):
    from sklearn.svm import SVR

    svr = SVR(C=1.0, epsilon=0.2)
    svr.fit(x_train, y_train.ravel())

    return svr.predict(x_test)


def SGD(x_train, y_train, x_test):
	sgd_mck = SGDClassifier()
	sgd_mck.fit(x_train, y_train.ravel())

	return sgd_mck.predict(x_test)


def score(y_predict, y):
    x = 0.0
    for i in range(0, y.__len__()):
        if y_predict[i] == y[i]:
            x = x + 1
    return x/y.__len__()