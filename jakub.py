import pandas as pd
import numpy as np
import openpyxl
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import ensemble
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix

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
def KNN(X_train, X_test, y_train, y_test, X_val, y_val, X, y):
    knn = KNeighborsClassifier(n_neighbors=50)  # n_neighbors=50 do wykresów    =2 do score
    knn.fit(X_train, y_train)
    knn_score = knn.score(X_test,y_test)
    knn_score_on_train = knn.score(X_train, y_train)
    knn_score_on_val = knn.score(X_val, y_val)
    knn_score_on_train_test = knn.score(X, y)
    #predictions_knn = knn.predict_proba(X_test)[::,1]
    #predictions_knn = knn.predict(X)
    predictions_knn = knn.predict_proba(X_val)[::,1]
    #predictions_knn = knn.predict(X_val)
    #fpr, tpr, _ = metrics.roc_curve(y_test, predictions_knn)
    #fpr, tpr, _ = metrics.roc_curve(y, predictions_knn)
    fpr, tpr, _ = metrics.roc_curve(y_val, predictions_knn)
    #auc = metrics.roc_auc_score(y_test, predictions_knn)
    #auc = metrics.roc_auc_score(y, predictions_knn)
    auc = metrics.roc_auc_score(y_val, predictions_knn)
    #print("KNN", confusion_matrix(y, predictions_knn))

    return knn_score, knn_score_on_train, knn_score_on_val, fpr, tpr, auc, knn_score_on_train_test


#Random Forest
def RF(X_train, X_test, y_train, y_test, X_val, y_val, X, y):
    rf = ensemble.RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)
    rf_score = rf.score(X_test, y_test)
    rf_score_on_train = rf.score(X_train, y_train)
    rf_score_on_val = rf.score(X_val, y_val)
    rf_score_on_train_test= rf.score(X, y)
    #predictions_rf = rf.predict_proba(X_test)[::,1]
    #predictions_rf = rf.predict(X)
    predictions_rf = rf.predict_proba(X_val)[::,1]
    #predictions_rf = rf.predict(X_val)
    #fpr, tpr, _ = metrics.roc_curve(y_test, predictions_rf)
    #fpr, tpr, _ = metrics.roc_curve(y, predictions_rf)
    fpr, tpr, _ = metrics.roc_curve(y_val, predictions_rf)
    #auc = metrics.roc_auc_score(y_test, predictions_rf)
    ##auc = metrics.roc_auc_score(y, predictions_rf)
    auc = metrics.roc_auc_score(y_val, predictions_rf)
    #print("RF",confusion_matrix(y, predictions_rf))

    return rf_score, rf_score_on_train, rf_score_on_val, fpr, tpr, auc, rf_score_on_train_test


#SVC
def SVC(X_train, X_test, y_train, y_test, X_val, y_val, X, y):
    from sklearn.svm import SVC
    svc = SVC(probability=True, kernel='linear')
    svc.fit(X_train, y_train)
    svc_score = svc.score(X_test, y_test)
    svc_score_on_train = svc.score(X_train, y_train)
    svc_score_on_val = svc.score(X_val, y_val)
    svc_score_on_train_test = svc.score(X, y)
    #predictions_svc= svc.predict_proba(X_test)[::,1]
    #predictions_svc= svc.predict(X)
    predictions_svc= svc.predict_proba(X_val)[::,1]
    #predictions_svc= svc.predict(X_val)
    #fpr, tpr, _ = metrics.roc_curve(y_test, predictions_svc)
    #fpr, tpr, _ = metrics.roc_curve(y, predictions_svc)
    fpr, tpr, _ = metrics.roc_curve(y_val, predictions_svc)
    #auc = metrics.roc_auc_score(y_test, predictions_svc)
    #auc = metrics.roc_auc_score(y, predictions_svc)
    auc = metrics.roc_auc_score(y_val, predictions_svc)
    #print("SVC",confusion_matrix(y, predictions_svc))

    return svc_score, svc_score_on_train, svc_score_on_val, fpr, tpr, auc, svc_score_on_train_test

#print(cross_val_score(svc, X, y, cv=10))

#Naive Bayes
def GNB(X_train, X_test, y_train, y_test, X_val, y_val, X, y):
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    gnb_score = gnb.score(X_test, y_test)
    gnb_score_on_train = gnb.score(X_train, y_train)
    gnb_score_on_val = gnb.score(X_val, y_val)
    gnb_score_on_train_test = gnb.score(X, y)
    #predictions_gnb = gnb.predict_proba(X_test)[::,1]
    #predictions_gnb = gnb.predict(X)
    predictions_gnb = gnb.predict_proba(X_val)[::,1]
    #predictions_gnb = gnb.predict(X_val)
    #fpr, tpr, _ = metrics.roc_curve(y_test, predictions_gnb)
    #fpr, tpr, _ = metrics.roc_curve(y, predictions_gnb)
    fpr, tpr, _ = metrics.roc_curve(y_val, predictions_gnb)
    #auc = metrics.roc_auc_score(y_test, predictions_gnb)
    #auc = metrics.roc_auc_score(y, predictions_gnb)
    auc = metrics.roc_auc_score(y_val, predictions_gnb)
    #print("GNB",confusion_matrix(y, predictions_gnb))

    return gnb_score, gnb_score_on_train, gnb_score_on_val, fpr, tpr, auc, gnb_score_on_train_test

#print(cross_val_score(gnb, X, y, cv=10))

def Wykres(fpr1, tpr1, auc1, fpr2, tpr2, auc2, method):
    plt.plot(fpr1, tpr1, label="70 30 AUC=" + str(auc1))
    plt.plot(fpr2, tpr2, label="10 90 AUC=" + str(auc2))
    # plt.plot(fpr2, tpr2, label="100 0")
    plt.title(method)
    plt.ylabel('True Positive Rage')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.show()

def Wykres_4_na_1(fpr1, tpr1, auc1, fpr2, tpr2, auc2, fpr3, tpr3, auc3, fpr4, tpr4, auc4, title):
    plt.plot(fpr1, tpr1, label="KNN AUC=" + str(auc1))
    plt.plot(fpr2, tpr2, label="RF AUC=" + str(auc2))
    plt.plot(fpr3, tpr3, label="SVC AUC=" + str(auc3))
    plt.plot(fpr4, tpr4, label="GNB AUC=" + str(auc4))
    # plt.plot(fpr2, tpr2, label="100 0")
    plt.title(title)
    plt.ylabel('True Positive Rage')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.show()


def Print_scores(knn_score_70_30, knn_score_on_train_70_30, rf_score_70_30, rf_score_on_train_70_30, svc_score_70_30,
                 svc_score_on_train_70_30, gnb_score_70_30, gnb_score_on_train_70_30, knn_score_10_90,
                 knn_score_on_train_10_90, rf_score_10_90, rf_score_on_train_10_90, svc_score_10_90,
                 svc_score_on_train_10_90, gnb_score_10_90, gnb_score_on_train_10_90,
                 knn_score_on_val_70_30, rf_score_on_val_70_30,
                 svc_score_on_val_70_30, gnb_score_on_val_70_30, knn_score_on_val_10_90, rf_score_on_val_10_90,
                 svc_score_on_val_10_90, gnb_score_on_val_10_90, knn_score_on_train_test70_30,
                 rf_score_on_train_test70_30, svc_score_on_train_test70_30, gnb_score_on_train_test70_30,
                 knn_score_on_train_test10_90, rf_score_on_train_test10_90, svc_score_on_train_test10_90,
                 gnb_score_on_train_test10_90):

    print("70_30:\nk_nearest_neighbors: ", knn_score_70_30,"\nrandom_forest: ", rf_score_70_30, "\nsupport_vector_classification: ",
          svc_score_70_30, "\nnaive_bayes: ", gnb_score_70_30,)
    print("\n\n10_90:\nk_nearest_neighbors: ", knn_score_10_90,"\nrandom_forest: ", rf_score_10_90, "\nsupport_vector_classification: ",
          svc_score_10_90, "\nnaive_bayes: ", gnb_score_10_90,)
    print("\n\n70_30_on_train:\nk_nearest_neighbors: ", knn_score_on_train_70_30, "\nrandom_forest: ", rf_score_on_train_70_30,
          "\nsupport_vector_classification: ", svc_score_on_train_70_30, "\nnaive_bayes: ", gnb_score_on_train_70_30, )
    print("\n\n10_90_on_train:\nk_nearest_neighbors: ", knn_score_on_train_10_90, "\nrandom_forest: ", rf_score_on_train_10_90,
          "\nsupport_vector_classification: ",svc_score_on_train_10_90, "\nnaive_bayes: ", gnb_score_on_train_10_90, )
    print("\n\n70_30_on_val:\nk_nearest_neighbors: ", knn_score_on_val_70_30, "\nrandom_forest: ",
          rf_score_on_val_70_30, "\nsupport_vector_classification: ", svc_score_on_val_70_30, "\nnaive_bayes: ", gnb_score_on_val_70_30)
    print("\n\n10_90_on_val:\nk_nearest_neighbors: ", knn_score_on_val_10_90, "\nrandom_forest: ",
          rf_score_on_val_10_90, "\nsupport_vector_classification: ", svc_score_on_val_10_90, "\nnaive_bayes: ", gnb_score_on_val_10_90)
    print("\n\n70_30_on_train+test:\nk_nearest_neighbors: ", knn_score_on_train_test70_30, "\nrandom_forest: ",
          rf_score_on_train_test70_30, "\nsupport_vector_classification: ", svc_score_on_train_test70_30, "\nnaive_bayes: ",
          gnb_score_on_train_test70_30)
    print("\n\n10_90_on_train+test:\nk_nearest_neighbors: ", knn_score_on_train_test10_90, "\nrandom_forest: ",
          rf_score_on_train_test10_90, "\nsupport_vector_classification: ", svc_score_on_train_test10_90, "\nnaive_bayes: ",
          gnb_score_on_train_test10_90)

def JG_wywołaj(df):
    X = df.values
    y = df['Y'].values
    X = np.delete(X, 0, axis=1)

    df_val = pd.read_excel(r'dane_do_walidacji_jakub.xlsx')#
    X_val = df_val.values
    y_val = df_val['Y'].values
    X_val = np.delete(X_val, 0, axis=1)#

    X70_train, X30_test, y70_train, y30_test = div70_30(X, y)
    #print("X_70_train",X70_train,"X30_test", X30_test,"y70_train", y70_train, len(y70_train),"y30_test", y30_test, len(y30_test))
    knn_score_70_30, knn_score_on_train_70_30, knn_score_on_val_70_30, fpr_knn_70_30, tpr_knn_70_30, auc_knn_70_30,\
        knn_score_on_train_test70_30 = KNN(X70_train, X30_test, y70_train, y30_test, X_val, y_val, X, y)
    rf_score_70_30, rf_score_on_train_70_30, rf_score_on_val_70_30, fpr_rf_70_30, tpr_rf_70_30, auc_rf_70_30, \
        rf_score_on_train_test70_30 = RF(X70_train, X30_test, y70_train, y30_test, X_val, y_val, X, y)
    svc_score_70_30, svc_score_on_train_70_30, svc_score_on_val_70_30, fpr_svc_70_30, tpr_svc_70_30, auc_svc_70_30,\
        svc_score_on_train_test70_30 = SVC(X70_train, X30_test, y70_train, y30_test, X_val, y_val, X, y)
    gnb_score_70_30, gnb_score_on_train_70_30, gnb_score_on_val_70_30, fpr_gnb_70_30, tpr_gnb_70_30, auc_gnb_70_30,\
        gnb_score_on_train_test70_30 = GNB(X70_train, X30_test, y70_train, y30_test, X_val, y_val, X, y)

    X10_train, X90_test, y10_train, y90_test = div10_90(X, y)
    knn_score_10_90, knn_score_on_train_10_90, knn_score_on_val_10_90, fpr_knn_10_90, tpr_knn_10_90, auc_knn_10_90,\
        knn_score_on_train_test10_90 = KNN(X10_train, X90_test, y10_train, y90_test, X_val, y_val, X, y)
    rf_score_10_90, rf_score_on_train_10_90, rf_score_on_val_10_90, fpr_rf_10_90, tpr_rf_10_90, auc_rf_10_90, \
        rf_score_on_train_test10_90 = RF(X10_train, X90_test, y10_train, y90_test, X_val, y_val, X, y)
    svc_score_10_90, svc_score_on_train_10_90, svc_score_on_val_10_90, fpr_svc_10_90, tpr_svc_10_90, auc_svc_10_90,\
        svc_score_on_train_test10_90 = SVC(X10_train, X90_test, y10_train, y90_test, X_val, y_val, X, y)
    gnb_score_10_90, gnb_score_on_train_10_90, gnb_score_on_val_10_90, fpr_gnb_10_90, tpr_gnb_10_90, auc_gnb_10_90,\
        gnb_score_on_train_test10_90 =GNB(X10_train, X90_test, y10_train, y90_test, X_val, y_val, X, y)


    Print_scores(knn_score_70_30, knn_score_on_train_70_30, rf_score_70_30, rf_score_on_train_70_30, svc_score_70_30,
                 svc_score_on_train_70_30, gnb_score_70_30, gnb_score_on_train_70_30, knn_score_10_90,
                 knn_score_on_train_10_90, rf_score_10_90, rf_score_on_train_10_90, svc_score_10_90,
                 svc_score_on_train_10_90, gnb_score_10_90, gnb_score_on_train_10_90, knn_score_on_val_70_30, rf_score_on_val_70_30,
                 svc_score_on_val_70_30, gnb_score_on_val_70_30, knn_score_on_val_10_90, rf_score_on_val_10_90,
                 svc_score_on_val_10_90, gnb_score_on_val_10_90, knn_score_on_train_test70_30,
                 rf_score_on_train_test70_30, svc_score_on_train_test70_30, gnb_score_on_train_test70_30,
                 knn_score_on_train_test10_90, rf_score_on_train_test10_90, svc_score_on_train_test10_90,
                 gnb_score_on_train_test10_90)


    Wykres(fpr_knn_70_30,tpr_knn_70_30, auc_knn_70_30, fpr_knn_10_90, tpr_knn_10_90, auc_knn_10_90, "K Nearest Neighbors")
    Wykres(fpr_rf_70_30, tpr_rf_70_30,auc_rf_70_30,fpr_rf_10_90, tpr_rf_10_90, auc_rf_10_90, "Random Forest")
    Wykres(fpr_svc_70_30,tpr_svc_70_30, auc_svc_70_30, fpr_svc_10_90, tpr_svc_10_90, auc_svc_10_90, "Support Vector Classification")
    Wykres(fpr_gnb_70_30, tpr_gnb_70_30, auc_gnb_70_30, fpr_gnb_10_90, tpr_gnb_10_90, auc_gnb_10_90, "Naive Bayes")

    Wykres_4_na_1(fpr_knn_70_30, tpr_knn_70_30, auc_knn_70_30, fpr_rf_70_30, tpr_rf_70_30, auc_rf_70_30, fpr_svc_70_30,
                  tpr_svc_70_30, auc_svc_70_30, fpr_gnb_70_30, tpr_gnb_70_30, auc_gnb_70_30, "70_30 na walidacyjnej")
    Wykres_4_na_1(fpr_knn_10_90, tpr_knn_10_90, auc_knn_10_90, fpr_rf_10_90, tpr_rf_10_90, auc_rf_10_90, fpr_svc_10_90,
                  tpr_svc_10_90, auc_svc_10_90, fpr_gnb_10_90, tpr_gnb_10_90, auc_gnb_10_90, "10_90 na walidacyjnej")

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


