import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

sm = pd.read_csv("sm.csv").to_numpy()
X = sm[:,3:]
y = sm[:,1:2]

scaler = MinMaxScaler()
y = scaler.fit_transform(y)
y = y.squeeze()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
regr = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)
print(regr.score(X_test, y_test))