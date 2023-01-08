from sklearn.model_selection import train_test_split
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

import pandas as pd

# Importing dataset
df = pd.read_excel("./data.xlsx")
# print(df)
x = df.iloc[:, 1:].values
y = df.iloc[:, :1].values

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=0
)

# DECISION TREE
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)
predictionsDecisionTree = clf.predict(X_test)
scoreDecisionTree = clf.score(X_test, y_test)

print(scoreDecisionTree)
print(predictionsDecisionTree)

# RANDOM FOREST
rf = RandomForestClassifier(max_depth=15, random_state=0)
rf.fit(X_train, y_train.ravel())
predictionsRandomForest = rf.predict(X_test)
scoreRandomForest = rf.score(X_test, y_test)

print(predictionsRandomForest)
print(scoreRandomForest)

# Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train.ravel())
predictionsNaiveBayes = gnb.predict(X_test)
scoreNaiveBayes = gnb.score(X_test, y_test)

print(predictionsNaiveBayes)
print(scoreNaiveBayes)

# Logistic regression
lg = LogisticRegression(random_state=0)
lg.fit(X_train, y_train.ravel())
predictionsLogisticRegression = lg.predict(X_test)
scoreLogisticRegression = lg.score(X_train, y_train)

print(predictionsLogisticRegression)
print(scoreLogisticRegression)
#print(lg.predict_proba(X_test))
