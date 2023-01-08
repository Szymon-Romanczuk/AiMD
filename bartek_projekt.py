import pandas as pd
import numpy as np


df = pd.read_excel(r'Dane2.xlsx')
#df = pd.DataFrame(data)

X= df.values
Y =df['Y'].values
X= np.delete(X,0,axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.3,train_size=0.7,random_state=0)


#KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train,y_train)
knn_score = knn.score(X_test,y_test)
print("KNN",knn_score)

from sklearn.model_selection import cross_val_score
print(cross_val_score(knn, X, Y, cv=10))


#LinearSVC
from sklearn.svm import SVC
svm = SVC(probability=True, kernel='linear')
svm.fit(X_train,y_train)
svm_score = svm.score(X_test, y_test)
print("SVN:", svm_score)

print(cross_val_score(svm, X, Y, cv=10))

#drzewo decyzyjne
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier(random_state=0, max_depth=2)
decision_tree = decision_tree.fit(X_test, y_test)
decision_tree_score = decision_tree.score(X_test,y_test)
print("decision_tree", decision_tree_score)

print(cross_val_score(decision_tree, X, Y, cv=10))


#LinearSVR
from sklearn.svm import SVR
svr =SVR(C=1.0, epsilon=0.2)
svr.fit(X_train,y_train)
svr_score = svr.score(X_test, y_test)
print("SVR:", svr_score)

print(cross_val_score(svr, X, Y, cv=10))

