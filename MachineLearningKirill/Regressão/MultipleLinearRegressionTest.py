#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 05:46:23 2022

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

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)

np.set_printoptions(precision = 2)
concat_result = np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)
concat_result = pd.DataFrame(concat_result)
concat_result.columns = ['Predict', 'Test']

from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
