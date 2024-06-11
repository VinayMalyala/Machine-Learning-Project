import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df = pd.read_csv("/content/bottle.csv")
print(df)


df_bin = df[['Salnty', 'T_degC']]
df_bin


df_bin.columns = ['Sal', 'Temp']
df_bin


sns.lmplot(x="Sal", y="Temp", data=df_bin)
sns.lmplot(x="Sal", y="Temp", data=df_bin, order=2, ci=None)



df_bin.fillna(method="ffill", inplace=True)
df_bin.dropna(inplace=True)



X = np.array(df_bin['Sal']).reshape(-1, 1)
Y = np.array(df_bin['Temp']).reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)



model = LinearRegression()
model.fit(X_train, y_train)
print(model.score(X_train, y_train))



df_bin_500 = df_bin[:][:500]
df_bin_500
sns.lmplot(x="Sal", y="Temp", data=df_bin_500, order=2, ci=None)



df_bin_500.fillna(method="ffill", inplace=True)
df_bin_500.dropna(inplace=True)



X = np.array(df_bin['Sal']).reshape(-1, 1)
Y = np.array(df_bin['Temp']).reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

model = LinearRegression()
model.fit(X_train, y_train)
print(model.score(X_train, y_train))