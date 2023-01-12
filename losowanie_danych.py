import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('./dane_do_analizy.csv', sep=';')
df_val = pd.read_csv('./grupa_werfikacyjna.csv', sep=';')

x = df.iloc[:, 1:].values
y = df.iloc[:, :1].values

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.9, random_state=8
)

x_val = df_val.iloc[:, 1:].values
y_val = df_val.iloc[:, :1].values
