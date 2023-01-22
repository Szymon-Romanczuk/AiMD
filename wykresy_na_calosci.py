import numpy as np
from sklearn.model_selection import train_test_split
from methods import *
from losowanie_danych import r10_90, r70_30
from matplotlib import pyplot as plt

#Tu podajesz dane wejściowe, są losowne w pliku "losowanie danch",
#można zrobić ile się chce różnych podziałów

x_train, x_test, x_val, y_train, y_test, y_val = r70_30()

#stworzenie modelu, te 3 są stworzone w pliku methods,
#można dodać nowe według schematu
#wystarczy spojrzeć na stworzone metody
lg = logistic_regression_model(x_train, y_train)
#dodajesz
plt.title("Wykresiki")
Wykres(lg, x_test, y_test, "regresja liniowa")

#stworzenie modelu
lg = random_forest_model(x_train, y_train)
#Wykres (model, dane_x, dane_y, podpis wykresu na legendzie_
Wykres(lg, x_test, y_test, "random forest")

lg = linear_discrimanant_analysis_model(x_train, y_train)
Wykres(lg, x_test, y_test, "klasyczna funkcja dyskryminacyjna")
#wyświetlneie wykresu
plt.show()

#w konsoli wyidać tabele prawdy dla poszczególnych modeli