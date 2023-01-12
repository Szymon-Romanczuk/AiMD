import pandas as pd
from sklearn.model_selection import train_test_split


def get_data(test_group, val_group):
    df = pd.read_csv('./Arkusz2.csv', sep=';')

    x = df.iloc[:, 1:].values
    y = df.iloc[:, :1].values

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_group, random_state=8
    )

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_group, random_state=8)

    return x_train, y_train, x_test, y_test, x_val, y_val
