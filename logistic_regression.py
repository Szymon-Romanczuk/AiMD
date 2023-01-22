from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as st


def generate_results_dataset(preds, ci):
  df = pd.DataFrame()
  df['prediction'] = preds
  if ci >= 0:
    df['upper'] = preds + ci
    df['lower'] = preds - ci
  else:
    df['upper'] = preds - ci
    df['lower'] = preds + ci

  return df


# df_all = pd.read_csv('./wszystkie_dane.csv', sep=';')
# df90 = pd.read_csv('./dane_do_analizy.csv', sep=';')
# df_val = pd.read_csv('./grupa_werfikacyjna.csv', sep=';')
#
# x_val = df_val.iloc[:, 1:].values
# y_val = df_val.iloc[:, :1].values


def data90():
  df = pd.read_csv('./dane_do_analizy.csv', sep=';')
  x = df.iloc[:, 1:].values
  y = df.iloc[:, :1].values
  return x, y


def validation_data():
  df_val = pd.read_csv('./grupa_werfikacyjna.csv', sep=';')
  x_val = df_val.iloc[:, 1:].values
  y_val = df_val.iloc[:, :1].values
  return x_val, y_val


def all_data():
  df_all = pd.read_csv('./wszystkie_dane.csv', sep=';')
  x = df_all.iloc[:, 1:].values
  y = df_all.iloc[:, :1].values
  return x, y


x, y = all_data()
x_t, y_t = data90()
x_val, y_val = validation_data()

# alpha = 0.05

x_train90, x_test10, y_train90, y_test10 = train_test_split(
  x_t, y_t, test_size=0.1, random_state=8
)


# lg = LogisticRegression(random_state=0)
# lin_model = lg.fit(x, y.ravel())

# predictions = lg.predict(x_val)

# log_cf = sm.Logit(y_train90, x_train90)
# classifier = log_cf.fit(x_t, y_t)
# y_pred = log_cf.predict(x_t, y_t)
# print(log_cf.data)

a = st.t.interval(confidence=0.95, df=x)
print('Confidace interval')
print(a)

def get_logistic_regression_score(x_train, y_train, x_test, y_test, x_validation, y_validation):
  lg = LogisticRegression(random_state=8, max_iter=1000000, solver='newton-cholesky')
  lg.fit(x_train, y_train.ravel())
  predictions_logistic_regression = lg.predict(x_test)
  print(lg.score(x_test, y_test))
  # print(lg.score(x_train, y_train))
  # print(lg.score(x_validation, y_validation))
  # print(lg.score(x_val, y_val))
  # print(confusion_matrix(y_test, predictions_logistic_regression))
  # print(confusion_matrix(y_train, predictions_logistic_regression))
  print(confusion_matrix(y_test, predictions_logistic_regression))
  print(lg.get_params())

# ucząca 100%, testowane na 10% (walidacyjna)
get_logistic_regression_score(x, y, x_val, y_val, x_val, y_val)
# ucząca 100%, testowane na 90% (bez walidacyjnej)
# get_logistic_regression_score(x, y, x_t, y_t, x_val, y_val)
# ucząca 90%, testowane na walidacyjnej
# get_logistic_regression_score(x_t, y_t, x_val, y_val, x, y)
# ucząca 10%, testowane na 10% walidacyjny
# get_logistic_regression_score(x_test10, y_test10, x_val, y_val, x, y)


# ucząca 100%, testowane na 90% (bez walidacyjnej)
# [[3192   35]
#  [  16 3237]]
# accuracy = 0.9921296296296296

# ucząca 100%, testowane na 10% (walidacyjna)
# [[369   4]
#  [  4 343]]
# accuracy = 0.9888888888888889
# {'C': 1.0, 'class_weight': None, 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1, 'l1_ratio': None, 'max_iter': 1000000, 'multi_class': 'auto', 'n_jobs': None, 'penalty': 'l2', 'random_state': 8, 'solver': 'lbfgs', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}

# ucząca 90%, testowane na 10% walidacyjny
# accuracy = 0.9861111111111112
# [[368   5]
#  [  5 342]]
# {'C': 1.0, 'class_weight': None, 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1, 'l1_ratio': None, 'max_iter': 1000000, 'multi_class': 'auto', 'n_jobs': None, 'penalty': 'l2', 'random_state': 8, 'solver': 'lbfgs', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}

# ucząca 10%, testowane na 10% walidacyjny
# 0.9763888888888889
# [[360  13]
#  [  4 343]]
# {'C': 1.0, 'class_weight': None, 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1, 'l1_ratio': None, 'max_iter': 1000000, 'multi_class': 'auto', 'n_jobs': None, 'penalty': 'l2', 'random_state': 8, 'solver': 'lbfgs', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}



def get_plots_for_logistic_regression():
  dt1 = LogisticRegression(random_state=8, max_iter=1000000)
  dt1.fit(x, y.ravel())
  predictions_decision_tree1 = dt1.predict_proba(x_val)[::, 1]
  fpr_dt1, trp_dt1, _ = metrics.roc_curve(y_val.ravel(), predictions_decision_tree1)
  auc_dt1 = metrics.roc_auc_score(y_val, predictions_decision_tree1)

  # dt2 = LogisticRegression(random_state=8)
  # dt2.fit(wojtek_train70, y_train70.ravel())
  # predictions_decision_tree2 = dt1.predict_proba(wojtek_test30)[::, 1]
  # fpr_dt2, trp_dt2, _ = metrics.roc_curve(y_test30.ravel(), predictions_decision_tree2)
  # auc_dt2 = metrics.roc_auc_score(y_test30, predictions_decision_tree2)

  plt.plot(fpr_dt1, trp_dt1, label="Train 100% test 10% (validation) AUC=" + str(auc_dt1))
  # plt.plot(fpr_dt2, trp_dt2, label="Train 70% AUC=" + str(auc_dt2))
  plt.title("Curve ROC Logistic Regression")
  plt.ylabel('True Positive Rate')
  plt.xlabel('False Positive Rate')
  plt.legend(loc=4)
  plt.show()































# residuals = y_train - lg.predict(x_train)
#
# ci = np.quantile(residuals, 0.95)
# preds = lg.predict(x_test)
#
# df = generate_results_dataset(preds, ci)
# print(df)

# print(lg.score(x_val, predictions))
# print(lg.coef_)

# the coefficients of the regression model
# # coefs = np.r_[[lg.intercept_], lg.coef_]
# coefs = np.array(lg.coef_)
# # build an auxiliary dataframe with the constant term in it
# X_aux = x_train.copy()
# X_aux.insert(0, 'const', 1)
# # degrees of freedom
# dof = -np.diff(X_aux.shape)[0]
# # Student's t-distribution table lookup
# t_val = stats.t.isf(alpha/2, dof)
# # MSE of the residuals
# mse = np.sum((y_train - lin_model.predict(x_train)) ** 2) / dof
# # inverse of the variance of the parameters
# var_params = np.diag(np.linalg.inv(X_aux.T.dot(X_aux)))
# # distance between lower and upper bound of CI
# gap = t_val * np.sqrt(mse * var_params)
#
# conf_int = pd.DataFrame({'lower': coefs - gap, 'upper': coefs + gap}, index=X_aux.columns)


# alpha = 0.05  # 95% confidence interval
# lr = sm.OLS(y, sm.add_constant(x)).fit()
# log = smf.logit(y, data=x)
# conf_interval = lr.conf_int(alpha)

# print(conf_interval)

# X2 = sm.add_constant(x)
# est = sm.OLS(y, X2)
# est2 = est.fit()
# p = est.predict(x_val)
# print(est2.summary())
# print(est.score(x_val, p))

# logit_model=sm.Logit(y, x)
# result=logit_model.fit()
# print(result.summary())
