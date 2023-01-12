import pandas as pd
import numpy as np
import openpyxl
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import ensemble
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

df = pd.read_excel(r'nowe_dane_jakub.xlsx')
#print(df)

# podział df
#X = df.values
#print("x:", X)
#y = df['Y'].values
#X = np.delete(X,0,axis=1)
#print("x po usunięciu", X)
#print("y:",y)



#podział danych na grupę testową i trenującą
def div70_30 (X, y):
    X70_train, X30_test, y70_train, y30_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=0)
    return X70_train, X30_test, y70_train, y30_test

def div10_90 (X, y):
    X10_train, X90_test, y10_train, y90_test = train_test_split(X, y, test_size=0.9, train_size=0.1, random_state=0)
    return X10_train, X90_test, y10_train, y90_test


#KNN
def KNN(X_train, X_test, y_train, y_test, X_val, y_val):
    knn = KNeighborsClassifier(n_neighbors=2)  # n_neighbors=10
    knn.fit(X_train, y_train)
    knn_score = knn.score(X_test,y_test)
    knn_score_on_train = knn.score(X_train, y_train)
    knn_score_on_val = knn.score(X_val, y_val)
    #print("KNN:", knn_score)
    return knn_score, knn_score_on_train, knn_score_on_val


#print(cross_val_score(knn, X, y, cv=10))  #cv ilość gróp na którew dzielimy... bierze 1 jako testową a reszte na uczące i tak 10 razy

#Random Forest
def RF(X_train, X_test, y_train, y_test, X_val, y_val):
    rf = ensemble.RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)
    rf_score = rf.score(X_test,y_test)
    rf_score_on_train = rf.score(X_train,y_train)
    rf_score_on_val = rf.score(X_val,y_val)
    #print("RF:", rf_score)
    return rf_score, rf_score_on_train, rf_score_on_val

#print(cross_val_score(rf, X, y, cv=10)) #walidacja krzyżowa

#SVC
def SVC(X_train, X_test, y_train, y_test, X_val, y_val):
    from sklearn.svm import SVC
    svc = SVC(probability=True, kernel='linear')
    svc.fit(X_train, y_train)
    svc_score = svc.score(X_test, y_test)
    svc_score_on_train = svc.score(X_train, y_train)
    svc_score_on_val = svc.score(X_val, y_val)
    #print("SVC:", svc_score)
    return svc_score, svc_score_on_train, svc_score_on_val

#print(cross_val_score(svc, X, y, cv=10))

#Naive Bayes
def GNB(X_train, X_test, y_train, y_test, X_val, y_val):
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    gnb_score = gnb.score(X_test, y_test)
    gnb_score_on_train = gnb.score(X_train, y_train)
    gnb_score_on_val = gnb.score(X_val, y_val)
    #print("GNB:", gnb_score)
    return gnb_score, gnb_score_on_train, gnb_score_on_val

#print(cross_val_score(gnb, X, y, cv=10))

def Print_scores(knn_score_70_30, knn_score_on_train_70_30, rf_score_70_30, rf_score_on_train_70_30, svc_score_70_30,
                 svc_score_on_train_70_30, gnb_score_70_30, gnb_score_on_train_70_30, knn_score_10_90,
                 knn_score_on_train_10_90, rf_score_10_90, rf_score_on_train_10_90, svc_score_10_90,
                 svc_score_on_train_10_90, gnb_score_10_90, gnb_score_on_train_10_90,
                 knn_score_on_val_70_30, rf_score_on_val_70_30,
                 svc_score_on_val_70_30, gnb_score_on_val_70_30, knn_score_on_val_10_90, rf_score_on_val_10_90,
                 svc_score_on_val_10_90, gnb_score_on_val_10_90):

    print("70_30:\nk_nearest_neighbors: ", knn_score_70_30,"\nrandom_forest: ", rf_score_70_30, "\nsupport_vector_classification: ",
          svc_score_70_30, "\nnaive_bayes: ", gnb_score_70_30,)
    print("\n\n10_90:\nk_nearest_neighbors: ", knn_score_10_90,"\nrandom_forest: ", rf_score_10_90, "\nsupport_vector_classification: ",
          svc_score_10_90, "\nnaive_bayes: ", gnb_score_10_90,)
    print("\n\n70_30_on_train:\nk_nearest_neighbors: ", knn_score_on_train_70_30, "\nrandom_forest: ", rf_score_on_train_70_30,
          "\nsupport_vector_classification: ", svc_score_on_train_70_30, "\nnaive_bayes: ", gnb_score_on_train_70_30, )
    print("\n\n10_90_on_train:\nk_nearest_neighbors: ", knn_score_on_train_10_90, "\nrandom_forest: ", rf_score_on_train_10_90,
          "\nsupport_vector_classification: ",svc_score_on_train_10_90, "\nnaive_bayes: ", gnb_score_on_train_10_90, )
    print("\n\n70_30_on_val:\nk_nearest_neighbors: ", knn_score_on_val_70_30, "\nrandom_forest: ",
         rf_score_on_val_70_30, "\nsupport_vector_classification: ", svc_score_on_val_70_30, "\nnaive_bayes: ", gnb_score_on_val_70_30, )
    print("\n\n10_90_on_val:\nk_nearest_neighbors: ", knn_score_on_val_10_90, "\nrandom_forest: ",
          rf_score_on_val_10_90, "\nsupport_vector_classification: ", svc_score_on_val_10_90, "\nnaive_bayes: ", gnb_score_on_val_10_90, )

def JG_wywołaj(df):
    X = df.values
    y = df['Y'].values
    X = np.delete(X, 0, axis=1)

    df_val = pd.read_excel(r'dane_do_walidacji_jakub.xlsx')#
    X_val = df_val.values#
    y_val = df_val['Y'].values#
    X_val = np.delete(X_val, 0, axis=1)#

    X70_train, X30_test, y70_train, y30_test = div70_30(X, y)
    #print("X_70_train",X70_train,"X30_test", X30_test,"y70_train", y70_train, len(y70_train),"y30_test", y30_test, len(y30_test))
    knn_score_70_30, knn_score_on_train_70_30, knn_score_on_val_70_30 = KNN(X70_train, X30_test, y70_train, y30_test, X_val, y_val)
    rf_score_70_30, rf_score_on_train_70_30, rf_score_on_val_70_30 = RF(X70_train, X30_test, y70_train, y30_test, X_val, y_val)
    svc_score_70_30, svc_score_on_train_70_30, svc_score_on_val_70_30 = SVC(X70_train, X30_test, y70_train, y30_test, X_val, y_val)
    gnb_score_70_30, gnb_score_on_train_70_30, gnb_score_on_val_70_30 = GNB(X70_train, X30_test, y70_train, y30_test, X_val, y_val)

    X10_train, X90_test, y10_train, y90_test = div10_90(X, y)
    knn_score_10_90, knn_score_on_train_10_90, knn_score_on_val_10_90 = KNN(X10_train, X90_test, y10_train, y90_test, X_val, y_val)
    rf_score_10_90, rf_score_on_train_10_90, rf_score_on_val_10_90 = RF(X10_train, X90_test, y10_train, y90_test, X_val, y_val)
    svc_score_10_90, svc_score_on_train_10_90, svc_score_on_val_10_90 = SVC(X10_train, X90_test, y10_train, y90_test, X_val, y_val)
    gnb_score_10_90, gnb_score_on_train_10_90, gnb_score_on_val_10_90 = GNB(X10_train, X90_test, y10_train, y90_test, X_val, y_val)

    Print_scores(knn_score_70_30, knn_score_on_train_70_30, rf_score_70_30, rf_score_on_train_70_30, svc_score_70_30,
                 svc_score_on_train_70_30, gnb_score_70_30, gnb_score_on_train_70_30, knn_score_10_90,
                 knn_score_on_train_10_90, rf_score_10_90, rf_score_on_train_10_90, svc_score_10_90,
                 svc_score_on_train_10_90, gnb_score_10_90, gnb_score_on_train_10_90, knn_score_on_val_70_30, rf_score_on_val_70_30,
                 svc_score_on_val_70_30, gnb_score_on_val_70_30, knn_score_on_val_10_90, rf_score_on_val_10_90,
                 svc_score_on_val_10_90, gnb_score_on_val_10_90)

    return {
        "70_30": {
            'k nearest neighbors': knn_score_70_30,
            'random_forest': rf_score_70_30,
            'support vector classification': svc_score_70_30,
            'naive_bayes': gnb_score_70_30,
        },
        "10_90": {
            'k nearest neighbors': knn_score_10_90,
            'random_forest': rf_score_10_90,
            'support vector classification': svc_score_10_90,
            'naive_bayes': gnb_score_10_90,
        },
        "70_30_on_train": {
            'k nearest neighbors': knn_score_on_train_70_30,
            'random_forest': rf_score_on_train_70_30,
            'support vector classification': svc_score_on_train_70_30,
            'naive_bayes': gnb_score_on_train_70_30,
        },
        "10_90_on_train": {
            'k nearest neighbors': knn_score_on_train_10_90,
            'random_forest': rf_score_on_train_10_90,
            'support vector classification': svc_score_on_train_10_90,
            'naive_bayes': gnb_score_on_train_10_90,
        },
        "70_30_on_val": {
            'k nearest neighbors': knn_score_on_val_70_30,
            'random_forest': rf_score_on_val_70_30,
            'support vector classification': svc_score_on_val_70_30,
            'naive_bayes': gnb_score_on_val_70_30,
        },
        "10_90_on_val": {
            'k nearest neighbors': knn_score_on_val_10_90,
            'random_forest': rf_score_on_val_10_90,
            'support vector classification': svc_score_on_val_10_90,
            'naive_bayes': gnb_score_on_val_10_90,
        }
    }

JG_wywołaj(df)
