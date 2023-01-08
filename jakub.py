import pandas as pd
import numpy as np
import openpyxl
from sklearn.model_selection import cross_val_score

#wczytywanie danych
#df = pd.read_excel(r'C:\Users\Kuba\Desktop\dane_do-_modelowania.xlsl', sep=';')
#print(df)
df = pd.read_excel(r'jakub.xlsx')
print(df)

# podział df
X = df.values
print("x:", X)
y = df['Y'].values
X = np.delete(X,0,axis=1)
print("x po usunięciu", X)
print("y:",y)

#podział danych na grupę testową i trenującą
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=0)

#sc.fit(X_train)
#X_train = sc.transform(X_train)
#X_test = sc.transform(X_test)

#KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=2)  # n_neighbors=10
knn.fit(X_train, y_train)
knn_score = knn.score(X_test,y_test)          #liczony jako accuracy
print("KNN:", knn_score)


print(cross_val_score(knn, X, y, cv=10))  #cv ilość gróp na którew dzielimy... bierze 1 jako testową a reszte na uczące i tak 10 razy

#Random Forest
from sklearn import ensemble
rf = ensemble.RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
rf_score = rf.score(X_test,y_test)
print("RF:", rf_score)

print(cross_val_score(rf, X, y, cv=10)) #walidacja krzyżowa

#SVC
from sklearn.svm import SVC
svc = SVC(probability=True, kernel='linear')
svc.fit(X_train, y_train)
svc_score = svc.score(X_test, y_test)
print("SVC:", svc_score)

print(cross_val_score(svc, X, y, cv=10))

#Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
gnb_score = gnb.score(X_test, y_test)
print("GNB:", gnb_score)

print(cross_val_score(gnb, X, y, cv=10))
