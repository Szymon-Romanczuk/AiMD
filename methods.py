from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.linear_model import SGDClassifier


def decision_tree(x_train, y_train):
    dt = DecisionTreeClassifier(random_state=0)
    dt.fit(x_train, y_train)

    return dt


def random_forest(x_train, y_train):
    rf = RandomForestClassifier(max_depth=15, random_state=0)
    rf.fit(x_train, y_train.ravel())

    return rf


def gaussian_naive_bayes(x_train, y_train):
    gnb = GaussianNB()
    gnb.fit(x_train, y_train.ravel())

    return gnb


def logistic_regression(x_train, y_train):
    lg = LogisticRegression(random_state=0)
    lg.fit(x_train, y_train.ravel())

    return lg

def KNN(x_train, y_train):
    knn = KNeighborsClassifier(n_neighbors=2)  # n_neighbors=10
    knn.fit(x_train, y_train)
    return knn

def SVC(x_train, y_train):
    svc = SVC(probability=True, kernel='linear')
    svc.fit(x_train, y_train)

    return svc


def linear_discriminant_analysis(x_train, y_train):
    clf = LinearDiscriminantAnalysis()
    clf.fit(x_train, y_train)

    return clf


def natural_network(x_train, y_train):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(9, 5, 3), random_state=1, max_iter=6000)
    clf.fit(x_train, y_train)

    return clf

def gradient_boosting(x_train, y_train):
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    clf.fit(x_train, y_train)

    return clf

def radius_neighbors(x_train, y_train):
    clf = RadiusNeighborsClassifier(radius=1.8)
    clf.fit(x_train, y_train)

    return clf

def svr(x_train, y_train):
	svr = SVR(C=1.0, epsilon=0.2)
	svr.fit(x_train, y_train)

	return svr

def SGD(x_train, y_train):
	sgd_mck = SGDClassifier()
	sgd_mck.fit(x_train, y_train)

	return sgd_mck