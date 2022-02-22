#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 05:15:10 2022

@author: daniel
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('/home/daniel/Documentos/Udemy/Machine Learning A-Z/Machine+Learning+A-Z+(Model+Selection)/Machine Learning A-Z (Model Selection)/Regression/Data.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=(0))

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x_train = sc_x.fit_transform(x_train)
y_train = sc_y.fit_transform(y_train.reshape(-1, 1))

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x_train, y_train)

x_test_predict = regressor.predict(sc_x.transform(x_test))
y_pred = sc_y.inverse_transform(x_test_predict.reshape(-1, 1))

np.set_printoptions(precision = 2)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
