# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 19:21:18 2020

@author: tsjlk
"""
# In[1]:
import pandas as pd

dataset = pd.read_csv('ToyotaCorolla.csv')
print(dataset.shape)

dataset = dataset[["Age_08_04", "KM", "Fuel_Type", "HP", "Automatic", 
              "Doors", "Quarterly_Tax", "Mfr_Guarantee", "Guarantee_Period", 
              "Airco", "Automatic_airco", "CD_Player", "Powered_Windows", 
              "Sport_Model", "Tow_Bar", "Price"]]

dataset = pd.get_dummies(dataset, columns=['Fuel_Type'], prefix = ['Fuel'])

X = dataset.loc[:, dataset.columns != 'Price']
y = dataset.loc[:, dataset.columns == 'Price']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# In[2]:

from keras.models import Sequential
from keras.layers import Dense
from math import sqrt

classifier = Sequential()

classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu', input_dim = 17))

classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mean_squared_error'])

classifier.fit(X_train, y_train, batch_size = 10, epochs = 20)

y_pred = classifier.predict(X_test)
a = ((y_test - y_pred) ** 2)
#print(sum(a))
rmse = sqrt(sum((y_test - y_pred) ** 2) / len(y_test))
print(rmse)