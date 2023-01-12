import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve
from sklearn import metrics
import matplotlib.pyplot as plt

data_mck=pd.read_excel(r'maciek_excel.xlsx')


def div70_30 (X, y):
    X70_train, X30_test, y70_train, y30_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=0)
    return X70_train, X30_test, y70_train, y30_test

def div10_90 (X, y):
    X10_train, X90_test, y10_train, y90_test = train_test_split(X, y, test_size=0.9, train_size=0.1, random_state=0)
    return X10_train, X90_test, y10_train, y90_test


#regresja logistyczna
def RL(X_train, X_test, y_train, y_test,X_val, y_val):
    log_reg_mck = LogisticRegression()
    log_reg_mck.fit(X_train, y_train)
    log_reg_score_mck=log_reg_mck.score(X_test,y_test)
    log_reg_score_on_train_mck=log_reg_mck.score(X_train, y_train)
    log_rec_score_on_val_mck=log_reg_mck.score(X_val,y_val)
    predictions_log_reg_mck = log_reg_mck.predict(X_test)
    fpr, tpr, _ = metrics.roc_curve(y_test, predictions_log_reg_mck)
    auc = metrics.roc_auc_score(y_test, predictions_log_reg_mck)
    return log_reg_score_mck,log_reg_score_on_train_mck,log_rec_score_on_val_mck,fpr, tpr, auc


 #Stochastic Gradient Descent
def SGD(X_train, X_test, y_train, y_test,X_val, y_val):
    sgd_mck = SGDClassifier()
    sgd_mck.fit(X_train, y_train)
    sgd_mck_score=sgd_mck.score(X_test,y_test)
    sgd_mck_on_train_score=sgd_mck.score(X_train, y_train)
    sgd_mck_on_val_score=sgd_mck.score(X_val,y_val)
    predictions_sgd_mck = sgd_mck.predict(X_test)
    fpr, tpr, _ = metrics.roc_curve(y_test, predictions_sgd_mck)
    auc = metrics.roc_auc_score(y_test, predictions_sgd_mck)
    return sgd_mck_score,sgd_mck_on_train_score,sgd_mck_on_val_score,fpr, tpr, auc


#siec neuronowa
def NN(X_train, X_test, y_train, y_test,X_val, y_val):
    neural_network_mck = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(9, 5, 3), random_state=1,max_iter=6000)
    neural_network_mck.fit(X_train,y_train)
    neural_network_score_mck = neural_network_mck.score(X_test,y_test)
    neural_network_on_train_score_mck=neural_network_mck.score(X_train, y_train)
    neural_network_on_val_score_mck=neural_network_mck.score(X_val, y_val)
    predictions_neural_network_mck = neural_network_mck.predict(X_test)
    fpr, tpr, _ = metrics.roc_curve(y_test, predictions_neural_network_mck)
    auc = metrics.roc_auc_score(y_test, predictions_neural_network_mck)
    return neural_network_score_mck,neural_network_on_train_score_mck,neural_network_on_val_score_mck, fpr, tpr, auc

#Gradient boosting
def gradient_boosting(X_train, X_test, y_train, y_test,X_val, y_val):
    gb_mck = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    gb_mck.fit(X_train, y_train.ravel())
    gb_mck_score_mck = gb_mck.score(X_test,y_test)
    gb_mck_on_train_score_mck=gb_mck.score(X_train, y_train)
    gb_mck_on_val_score_mck=gb_mck.score(X_val, y_val)
    predictions_gb_mck_mck = gb_mck.predict(X_test)
    fpr, tpr, _ = metrics.roc_curve(y_test, predictions_gb_mck_mck)
    auc = metrics.roc_auc_score(y_test, predictions_gb_mck_mck)
    return gb_mck_score_mck,gb_mck_on_train_score_mck,gb_mck_on_val_score_mck, fpr, tpr, auc


def Wykres(fpr1, tpr1, auc1, fpr2, tpr2, auc2, method):
    plt.plot(fpr1, tpr1, label="70 30 AUC=" + str(auc1))
    plt.plot(fpr2, tpr2, label="10 90 AUC=" + str(auc2))
    # plt.plot(fpr2, tpr2, label="100 0")
    plt.title(method)
    plt.ylabel('True Positive Rage')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.show()



def print_scores_maciek(logistic_regression_score_70_30, logistic_regression_score_on_train_70_30, Stochastic_Gradient_Descent_score_70_30, Stochastic_Gradient_score_on_train_70_30, neuron_network_score_70_30,
                 neuron_network_score_on_train_70_30, gradient_boosting_score_70_30, gradient_boosting_score_on_train_70_30, logistic_regression_score_10_90,
                 logistic_regression_score_on_train_10_90, Stochastic_Gradient_Descent_score_10_90, Stochastic_Gradient_Descent_score_on_train_10_90, neuron_network_score_10_90,
                 neuron_network_score_on_train_10_90, gradient_boosting_score_10_90, gradient_boosting_score_on_train_10_90,logistic_regression_score_on_val_70_30,Stochastic_Gradient_score_on_val_70_30,
                 neuron_network_score_on_val_70_30,gradient_boosting_score_on_val_70_30,logistic_regression_score_on_val_10_90,Stochastic_Gradient_Descent_score_on_val_10_90,neuron_network_score_on_val_10_90
                 ,gradient_boosting_score_on_val_10_90):


    print("70_30:\nLogistic Regression: ", logistic_regression_score_70_30,"\nSGD: ", Stochastic_Gradient_Descent_score_70_30, "\nNeuron network: ",
          neuron_network_score_70_30, "\nGradient Boosting ", gradient_boosting_score_70_30,)
    print("\n\n10_90:\nLogistic Regression: ", logistic_regression_score_10_90,"\nSGD: ", Stochastic_Gradient_Descent_score_10_90, "\nNeuron network: ",
          neuron_network_score_10_90, "\nGradient Boosting ", gradient_boosting_score_10_90,)
    print("\n\n70_30_on_train:\nLogistic Regression: ", logistic_regression_score_on_train_70_30, "\nSGD: ", Stochastic_Gradient_score_on_train_70_30,
          "\nNeuron network: ", neuron_network_score_on_train_70_30, "\nGradient Boosting", gradient_boosting_score_on_train_70_30, )
    print("\n\n10_90_on_train:\nLogistic Regression: ", logistic_regression_score_on_train_10_90, "\nSGD: ", Stochastic_Gradient_Descent_score_on_train_10_90,
          "\nNeuron network: ",neuron_network_score_on_train_10_90, "\nGradient Boosting ", gradient_boosting_score_on_train_10_90, )
    print("\n\n70_30_on_val:\nLogistic Regression: ", logistic_regression_score_on_val_70_30, "\nSGD: ", gradient_boosting_score_on_val_70_30,
          "\nNeuron network: ", neuron_network_score_on_val_70_30, "\nGradient Boosting", neuron_network_score_on_val_70_30, )
    print("\n\n10_90_on_val:\nLogistic Regression: ", logistic_regression_score_on_val_10_90, "\nSGD: ", Stochastic_Gradient_Descent_score_on_val_10_90,
          "\nNeuron network: ",neuron_network_score_on_val_10_90, "\nGradient Boosting ", gradient_boosting_score_on_val_10_90)

def Wyniki(data_mck):
    X = data_mck.values
    y = data_mck['Y'].values
    X = np.delete(X, 0, axis=1)

    df_val = pd.read_excel(r'dane_do_walidacji_maciek.xlsx')#
    X_val = df_val.values#
    y_val = df_val['Y'].values#
    X_val = np.delete(X_val, 0, axis=1)#

    X70_train, X30_test, y70_train, y30_test = div70_30(X, y)
    logistic_regression_score_70_30, logistic_regression_score_on_train_70_30,logistic_regression_score_on_val_70_30,fpr_lr_70_30,tpr_lr_70_30,auc_lr_70_30 = RL(X70_train, X30_test, y70_train, y30_test,X_val,y_val)
    Stochastic_Gradient_Descent_score_70_30, Stochastic_Gradient_score_on_train_70_30,Stochastic_Gradient_score_on_val_70_30,fpr_sgd_70_30,tpr_sgd_70_30,auc_sgd_70_30 = SGD(X70_train, X30_test, y70_train, y30_test,X_val,y_val)
    neuron_network_score_70_30, neuron_network_score_on_train_70_30,neuron_network_score_on_val_70_30,fpr_nn_70_30,tpr_nn_70_30,auc_nn_70_30 = NN(X70_train, X30_test, y70_train, y30_test,X_val,y_val)
    gradient_boosting_score_70_30, gradient_boosting_score_on_train_70_30,gradient_boosting_score_on_val_70_30,fpr_gb_70_30,tpr_gb_70_30,auc_gb_70_30 = gradient_boosting(X70_train, X30_test, y70_train, y30_test,X_val,y_val)

    X10_train, X90_test, y10_train, y90_test = div10_90(X, y)
    logistic_regression_score_10_90, logistic_regression_score_on_train_10_90,logistic_regression_score_on_val_10_90,fpr_lr_10_90,tpr_lr_10_90, auc_lr_10_90 = RL(X10_train, X90_test, y10_train, y90_test,X_val,y_val)
    Stochastic_Gradient_Descent_score_10_90, Stochastic_Gradient_Descent_score_on_train_10_90,Stochastic_Gradient_Descent_score_on_val_10_90,fpr_sgd_10_90, tpr_sgd_10_90,auc_sgd_10_90 = SGD(X10_train, X90_test, y10_train, y90_test,X_val,y_val)
    neuron_network_score_10_90, neuron_network_score_on_train_10_90,neuron_network_score_on_val_10_90, fpr_nn_10_90,tpr_nn_10_90, auc_nn_10_90 = NN(X10_train, X90_test, y10_train, y90_test,X_val,y_val)
    gradient_boosting_score_10_90, gradient_boosting_score_on_train_10_90,gradient_boosting_score_on_val_10_90, fpr_gb_10_90, tpr_gb_10_90, auc_gb_10_90 = gradient_boosting(X10_train, X90_test, y10_train, y90_test,X_val,y_val)

    print_scores_maciek(logistic_regression_score_70_30, logistic_regression_score_on_train_70_30, Stochastic_Gradient_Descent_score_70_30, Stochastic_Gradient_score_on_train_70_30, neuron_network_score_70_30,
                 neuron_network_score_on_train_70_30, gradient_boosting_score_70_30, gradient_boosting_score_on_train_70_30, logistic_regression_score_10_90,
                 logistic_regression_score_on_train_10_90, Stochastic_Gradient_Descent_score_10_90, Stochastic_Gradient_Descent_score_on_train_10_90, neuron_network_score_10_90,
                 neuron_network_score_on_train_10_90, gradient_boosting_score_10_90, gradient_boosting_score_on_train_10_90,logistic_regression_score_on_val_70_30,Stochastic_Gradient_score_on_val_70_30,
                 neuron_network_score_on_val_70_30,gradient_boosting_score_on_val_70_30, logistic_regression_score_on_val_10_90,Stochastic_Gradient_Descent_score_on_val_10_90,neuron_network_score_on_val_10_90
                 ,gradient_boosting_score_on_val_10_90)
    
    Wykres(fpr_lr_70_30,tpr_lr_70_30, auc_lr_70_30,fpr_lr_10_90,tpr_lr_10_90, auc_lr_10_90, "Logistic Regression")
    Wykres(fpr_sgd_70_30, tpr_sgd_70_30,auc_sgd_70_30,fpr_sgd_10_90, tpr_sgd_10_90,auc_sgd_10_90, "Stochastic Gradient Descent")
    Wykres(fpr_nn_70_30,tpr_nn_70_30, auc_nn_70_30, fpr_nn_10_90,tpr_nn_10_90, auc_nn_10_90, "Neural Network")
    Wykres(fpr_gb_70_30, tpr_gb_70_30, auc_gb_70_30, fpr_gb_10_90, tpr_gb_10_90, auc_gb_10_90, "Gradient Boosting")


    return {
        "70_30": {
            'logistic_regression': logistic_regression_score_70_30,
            'Stochastic_Gradient_Descent': Stochastic_Gradient_Descent_score_70_30,
            'neuron_network': neuron_network_score_70_30,
            'gradient_boosting': gradient_boosting_score_70_30,
        },
        "10_90": {
            'logistic_regression': logistic_regression_score_10_90,
            'Stochastic_Gradient_Descent': Stochastic_Gradient_Descent_score_10_90,
            'neuron_network': neuron_network_score_10_90,
            'gradient_boosting': gradient_boosting_score_10_90,
        },
        "70_30_on_train": {
            'logistic_regression': logistic_regression_score_on_train_70_30,
            'Stochastic_Gradient_Descent': Stochastic_Gradient_score_on_train_70_30,
            'neuron_network': neuron_network_score_on_train_70_30,
            'gradient_boosting': gradient_boosting_score_on_train_70_30,
        },
        "10_90_on_train": {
            'logistic_regression': logistic_regression_score_on_train_10_90,
            'Stochastic_Gradient_Descent': Stochastic_Gradient_Descent_score_on_train_10_90,
            'neuron_network': neuron_network_score_on_train_10_90,
            'gradient_boosting': gradient_boosting_score_on_train_10_90,
        },
         "70_30_on_val": {
            'logistic_regression': logistic_regression_score_on_val_70_30,
            'Stochastic_Gradient_Descent': Stochastic_Gradient_score_on_val_70_30,
            'neuron_network': neuron_network_score_on_val_70_30,
            'gradient_boosting': gradient_boosting_score_on_val_70_30,
        "10_90_on_val": {
            'logistic_regression': logistic_regression_score_on_val_10_90,
            'Stochastic_Gradient_Descent': Stochastic_Gradient_Descent_score_on_val_10_90,
            'neuron_network': neuron_network_score_on_val_10_90,
            'gradient_boosting': gradient_boosting_score_on_val_10_90,
        }
    }
    }

Wyniki(data_mck)
