#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 06:23:09 2022

@author: daniel
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('/home/daniel/Documentos/Udemy/Machine Learning A-Z/Machine+Learning+A-Z+(Model+Selection)/Machine Learning A-Z (Model Selection)/Regression/Data.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=(0))

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
regressor_poly = PolynomialFeatures(degree=4)
x_train_poly = regressor_poly.fit_transform(x_train)
regressor = LinearRegression()
regressor.fit(x_train_poly, y_train)

y_pred = regressor.predict(regressor_poly.transform(x_test))

np.set_printoptions(precision = 2)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
