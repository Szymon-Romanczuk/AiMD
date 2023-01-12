import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

data_mck=pd.read_excel(r'maciek_excel.xlsx')
# X= data_mck.values
# Y =data_mck['Y'].values
# X= np.delete(X,0,axis=1)

def div70_30 (X, y):
    X70_train, X30_test, y70_train, y30_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=0)
    return X70_train, X30_test, y70_train, y30_test

def div10_90 (X, y):
    X10_train, X90_test, y10_train, y90_test = train_test_split(X, y, test_size=0.9, train_size=0.1, random_state=0)
    return X10_train, X90_test, y10_train, y90_test


#regresja logistyczna
def RL(X_train, y_train):
    log_reg_mck = LogisticRegression(random_state=0)
    log_reg_mck.fit(X_train, y_train)

    return log_reg_mck


 #Stochastic Gradient Descent
def SGD(X_train, y_train):
    sgd_mck = SGDClassifier()
    sgd_mck.fit(X_train, y_train)
    return sgd_mck

#siec neuronowa
def NN(X_train, y_train):
    neuron_network_mck = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(9, 5, 3), random_state=1,max_iter=6000)
    neuron_network_mck.fit(X_train,y_train)
    return neuron_network_mck

#AUC ROC CURVE
def AUC_ROC(X_train, X_test, y_train, y_test):
    log_reg = LogisticRegression()
    log_reg2 = LogisticRegression()
    log_reg.fit(X_train, y_train)
    log_reg2.fit(X_train, y_train)
    pred_prob1 = log_reg.predict_proba(X_test)
    pred_prob2 = log_reg2.predict_proba(X_train)
    # fpr1, tpr1, thresh1 = roc_curve(y_test, pred_prob1[:,1], pos_label=1)
    # random_probs = [0 for i in range(len(y_test))]
    # p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)
    auc_roc_score1 = roc_auc_score(y_test, pred_prob1[:,1])
    auc_roc_score2 = roc_auc_score(y_train, pred_prob2[:,1])
    #print("AUC - Roc Curve: ",auc_roc_score1)
    return auc_roc_score1,auc_roc_score2

def print_scores_maciek(logistic_regression_score_70_30, logistic_regression_score_on_train_70_30, Stochastic_Gradient_Descent_score_70_30, Stochastic_Gradient_score_on_train_70_30, neuron_network_score_70_30,
                 neuron_network_score_on_train_70_30, auc_roc_score_70_30, auc_roc_score_on_train_70_30, logistic_regression_score_10_90,
                 logistic_regression_score_on_train_10_90, Stochastic_Gradient_Descent_score_10_90, Stochastic_Gradient_Descent_score_on_train_10_90, neuron_network_score_10_90,
                 neuron_network_score_on_train_10_90, auc_roc_score_10_90, auc_roc_score_on_train_10_90):
    print("70_30:\nLogistic Regression: ", logistic_regression_score_70_30,"\nSGD: ", Stochastic_Gradient_Descent_score_70_30, "\nNeuron network: ",
          auc_roc_score_70_30, "\nAUC-Roc Curve: ", auc_roc_score_70_30,)
    print("\n\n10_90:\nLogistic Regression: ", logistic_regression_score_10_90,"\nSGD: ", Stochastic_Gradient_Descent_score_10_90, "\nNeuron network: ",
          neuron_network_score_10_90, "\nAUC-Roc Curve: ", auc_roc_score_10_90,)
    print("\n\n70_30_on_train:\nLogistic Regression: ", logistic_regression_score_on_train_70_30, "\nSGD: ", neuron_network_score_on_train_70_30,
          "\nNeuron network: ", neuron_network_score_on_train_70_30, "\nAUC-Roc Curve:", neuron_network_score_on_train_70_30, )
    print("\n\n10_90_on_train:\nLogistic Regression: ", logistic_regression_score_10_90, "\nSGD: ", Stochastic_Gradient_Descent_score_on_train_10_90,
          "\nNeuron network: ",neuron_network_score_on_train_10_90, "\nAUC-Roc Curve: ", auc_roc_score_on_train_10_90, )

def Wyniki(data_mck):
    X = data_mck.values
    y = data_mck['Y'].values
    X = np.delete(X, 0, axis=1)

    X70_train, X30_test, y70_train, y30_test = div70_30(X, y)
    logistic_regression_score_70_30, logistic_regression_score_on_train_70_30 = RL(X70_train, X30_test, y70_train, y30_test)
    Stochastic_Gradient_Descent_score_70_30, Stochastic_Gradient_score_on_train_70_30 = SGD(X70_train, X30_test, y70_train, y30_test)
    neuron_network_score_70_30, neuron_network_score_on_train_70_30 = NN(X70_train, X30_test, y70_train, y30_test)
    auc_roc_score_70_30, auc_roc_score_on_train_70_30 = AUC_ROC(X70_train, X30_test, y70_train, y30_test)

    X10_train, X90_test, y10_train, y90_test = div10_90(X, y)
    logistic_regression_score_10_90, logistic_regression_score_on_train_10_90 = RL(X10_train, X90_test, y10_train, y90_test)
    Stochastic_Gradient_Descent_score_10_90, Stochastic_Gradient_Descent_score_on_train_10_90 = SGD(X10_train, X90_test, y10_train, y90_test)
    neuron_network_score_10_90, neuron_network_score_on_train_10_90 = NN(X10_train, X90_test, y10_train, y90_test)
    auc_roc_score_10_90, auc_roc_score_on_train_10_90 = AUC_ROC(X10_train, X90_test, y10_train, y90_test)

    print_scores_maciek(logistic_regression_score_70_30, logistic_regression_score_on_train_70_30, Stochastic_Gradient_Descent_score_70_30, Stochastic_Gradient_score_on_train_70_30, neuron_network_score_70_30,
                 neuron_network_score_on_train_70_30, auc_roc_score_70_30, auc_roc_score_on_train_70_30, logistic_regression_score_10_90,
                 logistic_regression_score_on_train_10_90, Stochastic_Gradient_Descent_score_10_90, Stochastic_Gradient_Descent_score_on_train_10_90, neuron_network_score_10_90,
                 neuron_network_score_on_train_10_90, auc_roc_score_10_90, auc_roc_score_on_train_10_90)

    return {
        "70_30": {
            'logistic_regression': logistic_regression_score_70_30,
            'Stochastic_Gradient_Descent': Stochastic_Gradient_Descent_score_70_30,
            'neuron_network': neuron_network_score_70_30,
            'auc_roc': auc_roc_score_70_30,
        },
        "10_90": {
            'logistic_regression': logistic_regression_score_10_90,
            'Stochastic_Gradient_Descent': Stochastic_Gradient_Descent_score_10_90,
            'neuron_network': neuron_network_score_10_90,
            'auc_roc': auc_roc_score_10_90,
        },
        "70_30_on_train": {
            'logistic_regression': logistic_regression_score_on_train_70_30,
            'Stochastic_Gradient_Descent': Stochastic_Gradient_score_on_train_70_30,
            'neuron_network': neuron_network_score_on_train_70_30,
            'auc_roc': auc_roc_score_on_train_70_30,
        },
        "10_90_on_train": {
            'logistic_regression': logistic_regression_score_on_train_10_90,
            'Stochastic_Gradient_Descent': Stochastic_Gradient_Descent_score_on_train_10_90,
            'neuron_network': neuron_network_score_on_train_10_90,
            'auc_roc': auc_roc_score_on_train_10_90,
        }
    }

Wyniki(data_mck)
