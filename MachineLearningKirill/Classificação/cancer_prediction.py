#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 17:22:52 2022

@author: daniel
"""
#importing the libraries
import pandas as pd 

#importing the dataset
dataset = pd.read_csv('/home/daniel/Downloads/Logistic_Regression/Final Folder/Dataset/breast_cancer.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

#splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x , y, test_size=0.2 ,random_state=0)

#training the Logistic Regression model
from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression(random_state=0)
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)

#making the confusion matrix 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#accuracy cross validation
from sklearn.model_selection import cross_val_score
cross_score = cross_val_score(estimator = regressor, X = x_train, y = y_train, cv = 10)
print(f'The mean of accuracy is: {round((cross_score.mean()*100), 1)}' + '%')
print(f'The standart deviation is: {round((cross_score.std()*100), 2)}' + '%')
