import pandas as pd
import numpy as np
import openpyxl
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVR
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


df = pd.read_excel(r'Bartek_dane.xlsx')

#podział danych na grupę testową i trenującą
def div70_30 (X, y):
    X70_train, X30_test, y70_train, y30_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=0)
    return X70_train, X30_test, y70_train, y30_test

def div10_90 (X, y):
    X10_train, X90_test, y10_train, y90_test = train_test_split(X, y, test_size=0.9, train_size=0.1, random_state=0)
    return X10_train, X90_test, y10_train, y90_test


#KNN
def KNN(X_train, X_test, y_train, y_test, X_val, y_val):
    knn = KNeighborsClassifier(n_neighbors=50)  # n_neighbors=2
    knn.fit(X_train, y_train)
    knn_score = knn.score(X_test,y_test)
    knn_score_on_train = knn.score(X_train, y_train)
    knn_score_on_val = knn.score(X_val, y_val)
    predictions_knn = knn.predict(X_val)
    fpr, tpr, _ = metrics.roc_curve(y_val, predictions_knn)
    auc = metrics.roc_auc_score(y_val, predictions_knn)
    
    print("KNN nowe dane", confusion_matrix(y_val, predictions_knn))

    return knn_score, knn_score_on_train, knn_score_on_val, fpr, tpr, auc



#LinearSVC
def SVC(X_train, X_test, y_train, y_test, X_val, y_val):
    from sklearn.svm import SVC
    svc = SVC(probability=True, kernel='linear')
    svc.fit(X_train, y_train)
    svc_score = svc.score(X_test, y_test)
    svc_score_on_train = svc.score(X_train, y_train)
    svc_score_on_val = svc.score(X_val, y_val)
    predictions_svc= svc.predict(X_val)
    fpr, tpr, _ = metrics.roc_curve(y_val, predictions_svc)
    auc = metrics.roc_auc_score(y_val, predictions_svc)

    print("SVC nowe dane",confusion_matrix(y_val, predictions_svc))
    return svc_score, svc_score_on_train, svc_score_on_val, fpr, tpr, auc

#drzewo decyzyjne
def DecisionTree(X_train, X_test, y_train, y_test, X_val, y_val):
    decision_tree = DecisionTreeClassifier(random_state=0, max_depth=30)
    decision_tree = decision_tree.fit(X_test, y_test)
    decision_tree_score = decision_tree.score(X_test,y_test)
    decision_tree_score_on_train = decision_tree.score(X_train,y_train)
    decision_tree_score_on_val = decision_tree.score(X_val,y_val)
    prediction_decision_tree= decision_tree.predict(X_val)
    fpr, tpr, _ = metrics.roc_curve(y_val, prediction_decision_tree)
    auc = metrics.roc_auc_score(y_val, prediction_decision_tree)
    
    print("DT nowe dane",confusion_matrix(y_val, prediction_decision_tree))

    return decision_tree_score, decision_tree_score_on_train, decision_tree_score_on_val, fpr, tpr, auc



#Linear Discriminant Analysis
def LDA(X_train, X_test, y_train, y_test, X_val, y_val):
    clf = LinearDiscriminantAnalysis()
    clf.fit(X_train,y_train)
    clf_score = clf.score(X_test, y_test)
    clf_score_on_train = clf.score(X_train, y_train)
    clf_score_on_val = clf.score(X_val, y_val)
    predictions_clf = clf.predict(X_val)
    fpr, tpr, _ = metrics.roc_curve(y_val, predictions_clf)
    auc = metrics.roc_auc_score(y_val, predictions_clf)
    
    print("LDA nowe dane", confusion_matrix(y_val, predictions_clf))
    return clf_score, clf_score_on_train, clf_score_on_val, fpr, tpr, auc

def Wykres(fpr1, tpr1, auc1, fpr2, tpr2, auc2, method):
    plt.plot(fpr1, tpr1, label="70 30 AUC=" + str(auc1))
    plt.plot(fpr2, tpr2, label="10 90 AUC=" + str(auc2))
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

def Print_scores(knn_score_70_30, knn_score_on_train_70_30, decision_tree_score_70_30, decision_tree_score_on_train_70_30, svc_score_70_30,
                 svc_score_on_train_70_30, svr_score_70_30, svr_score_on_train_70_30, knn_score_10_90,
                 knn_score_on_train_10_90, svr_score_10_90, svr_score_on_train_10_90, svc_score_10_90,
                 svc_score_on_train_10_90, decision_tree_score_10_90, decision_tree_score_on_train_10_90,
                 knn_score_on_val_70_30, decision_tree_score_on_val_70_30,
                 svc_score_on_val_70_30, svr_score_on_val_70_30, knn_score_on_val_10_90, svr_score_on_val_10_90,
                 svc_score_on_val_10_90, decision_tree_score_on_val_10_90):

    print("70_30:\nk_nearest_neighbors: ", knn_score_70_30,"\ndecision_tree: ", decision_tree_score_70_30, "\nsupport_vector_classification: ",
          svc_score_70_30, "\nLinear Discriminant Analysis: ", svr_score_70_30,)
    print("\n\n10_90:\nk_nearest_neighbors: ", knn_score_10_90,"\ndecision_tree: ", decision_tree_score_10_90, "\nsupport_vector_classification: ",
          svc_score_10_90, "\nLinear Discriminant Analysis: ", svr_score_10_90,)
    print("\n\n70_30_on_train:\nk_nearest_neighbors: ", knn_score_on_train_70_30, "\ndecision_tree: ", decision_tree_score_on_train_70_30,
          "\nsupport_vector_classification: ", svc_score_on_train_70_30, "\nLinear Discriminant Analysis: ", svr_score_on_train_70_30, )
    print("\n\n10_90_on_train:\nk_nearest_neighbors: ", knn_score_on_train_10_90, "\ndecision_tree: ", decision_tree_score_on_train_10_90,
          "\nsupport_vector_classification: ",svc_score_on_train_10_90, "\nLinear Discriminant Analysis: ", svr_score_on_train_10_90, )
    print("\n\n70_30_on_val:\nk_nearest_neighbors: ", knn_score_on_val_70_30, "\ndecision_tree: ",
         decision_tree_score_on_val_70_30, "\nsupport_vector_classification: ", svc_score_on_val_70_30, "\nLinear Discriminant Analysis: ", svr_score_on_val_70_30, )
    print("\n\n10_90_on_val:\nk_nearest_neighbors: ", knn_score_on_val_10_90, "\ndecision_tree: ",
          decision_tree_score_on_val_10_90, "\nsupport_vector_classification: ", svc_score_on_val_10_90, "\nLinear Discriminant Analysis: ", svr_score_on_val_10_90, )

def BG_wywołaj(df):
    X = df.values
    y = df['Y'].values
    X = np.delete(X, 0, axis=1)

    df_val = pd.read_excel(r'Dane_do_walidacji_Bartek.xlsx')
    X_val = df_val.values#
    y_val = df_val['Y'].values#
    X_val = np.delete(X_val, 0, axis=1)#

    X70_train, X30_test, y70_train, y30_test = div70_30(X, y)
    #print("X_70_train",X70_train,"X30_test", X30_test,"y70_train", y70_train, len(y70_train),"y30_test", y30_test, len(y30_test))
    knn_score_70_30, knn_score_on_train_70_30, knn_score_on_val_70_30, fpr_knn_70_30, tpr_knn_70_30, auc_knn_70_30 =\
        KNN(X70_train, X30_test, y70_train, y30_test, X_val, y_val)
    decision_tree_score_70_30, decision_tree_score_on_train_70_30, decision_tree_score_on_val_70_30, fpr_decision_tree_70_30, tpr_decision_tree_70_30, auc_decision_tree_70_30 =\
        DecisionTree(X70_train, X30_test, y70_train, y30_test, X_val, y_val)
    svc_score_70_30, svc_score_on_train_70_30, svc_score_on_val_70_30, fpr_svc_70_30, tpr_svc_70_30, auc_svc_70_30 =\
        SVC(X70_train, X30_test, y70_train, y30_test, X_val, y_val)
    svr_score_70_30, svr_score_on_train_70_30, svr_score_on_val_70_30, fpr_svr_70_30, tpr_svr_70_30, auc_svr_70_30 =\
        LDA(X70_train, X30_test, y70_train, y30_test, X_val, y_val)

    X10_train, X90_test, y10_train, y90_test = div10_90(X, y)
    knn_score_10_90, knn_score_on_train_10_90, knn_score_on_val_10_90, fpr_knn_10_90, tpr_knn_10_90, auc_knn_10_90\
        = KNN(X10_train, X90_test, y10_train, y90_test, X_val, y_val)
    decision_tree_score_10_90, decision_tree_score_on_train_10_90, decision_tree_score_on_val_10_90, fpr_decision_tree_10_90, tpr_decision_tree_10_90, auc_decision_tree_10_90 =\
        DecisionTree(X10_train, X90_test, y10_train, y90_test, X_val, y_val)
    svc_score_10_90, svc_score_on_train_10_90, svc_score_on_val_10_90, fpr_svc_10_90, tpr_svc_10_90, auc_svc_10_90 =\
        SVC(X10_train, X90_test, y10_train, y90_test, X_val, y_val)
    svr_score_10_90, svr_score_on_train_10_90, svr_score_on_val_10_90, fpr_svr_10_90, tpr_svr_10_90, auc_svr_10_90 =\
        LDA(X10_train, X90_test, y10_train, y90_test, X_val, y_val)


    Print_scores(knn_score_70_30, knn_score_on_train_70_30, decision_tree_score_70_30, decision_tree_score_on_train_70_30, svc_score_70_30,
                 svc_score_on_train_70_30, svr_score_70_30, svr_score_on_train_70_30, knn_score_10_90,
                 knn_score_on_train_10_90, decision_tree_score_10_90, decision_tree_score_on_train_10_90, svc_score_10_90,
                 svc_score_on_train_10_90, svr_score_10_90, svr_score_on_train_10_90, knn_score_on_val_70_30, decision_tree_score_on_val_70_30,
                 svc_score_on_val_70_30, svr_score_on_val_70_30, knn_score_on_val_10_90, decision_tree_score_on_val_10_90,
                 svc_score_on_val_10_90, svr_score_on_val_10_90)


    Wykres(fpr_knn_70_30,tpr_knn_70_30, auc_knn_70_30, fpr_knn_10_90, tpr_knn_10_90, auc_knn_10_90, "K Nearest Neighbors")
    Wykres(fpr_decision_tree_70_30, tpr_decision_tree_70_30,auc_decision_tree_70_30,fpr_decision_tree_10_90,
           tpr_decision_tree_10_90, auc_decision_tree_10_90, "Decision tree")
    Wykres(fpr_svc_70_30,tpr_svc_70_30, auc_svc_70_30, fpr_svc_10_90, tpr_svc_10_90, auc_svc_10_90, "Support Vector Classification")
    Wykres(fpr_svr_70_30, tpr_svr_70_30, auc_svr_70_30, fpr_svr_10_90, tpr_svr_10_90, auc_svr_10_90, "Linear Discriminant Analysis")

    Wykres_4_na_1(fpr_knn_70_30,tpr_knn_70_30, auc_knn_70_30, fpr_decision_tree_70_30, tpr_decision_tree_70_30,auc_decision_tree_10_90,fpr_svc_70_30,tpr_svc_70_30, auc_svc_70_30,fpr_svr_70_30, tpr_svr_70_30, auc_svr_70_30,"70 30")
    Wykres_4_na_1(fpr_knn_10_90, tpr_knn_10_90, auc_knn_10_90,fpr_decision_tree_10_90,
           tpr_decision_tree_10_90, auc_decision_tree_10_90,fpr_svc_10_90, tpr_svc_10_90, auc_svc_10_90,fpr_svr_10_90, tpr_svr_10_90, auc_svr_10_90,"90 10")

    return {
        "70_30": {
            'k nearest neighbors': knn_score_70_30,
            'decision_tree': decision_tree_score_70_30,
            'support vector classification': svc_score_70_30,
            'Linear Discriminant Analysis': svr_score_70_30,
        },
        "10_90": {
            'k nearest neighbors': knn_score_10_90,
            'decision_tree': decision_tree_score_10_90,
            'support vector classification': svc_score_10_90,
            'Linear Discriminant Analysis': svr_score_10_90,
        },
        "70_30_on_train": {
            'k nearest neighbors': knn_score_on_train_70_30,
            'decision_tree': decision_tree_score_on_train_70_30,
            'support vector classification': svc_score_on_train_70_30,
            'svr': svr_score_on_train_70_30,
        },
        "10_90_on_train": {
            'k nearest neighbors': knn_score_on_train_10_90,
            'rdecision_tree': decision_tree_score_on_train_10_90,
            'support vector classification': svc_score_on_train_10_90,
            'Linear Discriminant Analysis': svr_score_on_train_10_90,
        },
        "70_30_on_val": {
            'k nearest neighbors': knn_score_on_val_70_30,
            'decision_tree': decision_tree_score_on_val_70_30,
            'support vector classification': svc_score_on_val_70_30,
            'Linear Discriminant Analysis': svr_score_on_val_70_30,
        },
        "10_90_on_val": {
            'k nearest neighbors': knn_score_on_val_10_90,
            'decision_tree': decision_tree_score_on_val_10_90,
            'support vector classification': svc_score_on_val_10_90,
            'Linear Discriminant Analysis': svr_score_on_val_10_90,
        }
    }

BG_wywołaj(df)