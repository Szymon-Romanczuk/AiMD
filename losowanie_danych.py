import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

df = pd.read_csv('./dane_do_analizy.csv', sep=';')
df_val = pd.read_csv('./grupa_werfikacyjna.csv', sep=';')

x = df.iloc[:, 1:].values
y = df.iloc[:, :1].values


def r70_30():
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=8
    )

    x_val = df_val.iloc[:, 1:].values
    y_val = df_val.iloc[:, :1].values
    return x_train, x_test, x_val, y_train, y_test, y_val


def r10_90():
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.9, random_state=8
    )

    x_val = df_val.iloc[:, 1:].values
    y_val = df_val.iloc[:, :1].values
    return x_train, x_test, x_val, y_train, y_test, y_val


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.9, random_state=8
)

x_val = df_val.iloc[:, 1:].values
y_val = df_val.iloc[:, :1].values

np_data = np.c_[np.array(y_train), np.array(x_train)]
df = pd.DataFrame(np_data)
filepath = 'train.xlsx'
df.to_excel(filepath, index=False)

np_data = np.c_[np.array(y_val), np.array(x_val)]
df = pd.DataFrame(np_data)
filepath = 'val.xlsx'
df.to_excel(filepath, index=False)

np_data = np.c_[np.array(y_test), np.array(x_test)]
df = pd.DataFrame(np_data)
filepath = 'test.xlsx'
df.to_excel(filepath, index=False)
