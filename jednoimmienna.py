import numpy as np
from sklearn.model_selection import train_test_split
from methods import *
from losowanie_danych import r10_90, r70_30
from matplotlib import pyplot as plt

x_train, x_test, x_val, y_train, y_test, y_val = r10_90()
jednoimmiena = random_forest(x_train, y_train, x_test)
print(score(jednoimmiena, y_test))
jednoimmiena = random_forest(x_train, y_train, x_val)
print(score(jednoimmiena, y_val))
