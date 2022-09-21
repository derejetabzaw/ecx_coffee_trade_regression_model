#!/usr/bin/python
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
import os 
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from numpy import argmax
import pickle 
from sklearn.metrics import accuracy_score


path = "Coffee_trade_data.csv"
data = pd.read_csv(path)



date_time = (pd.to_datetime(data['Trade Date'])).dt.strftime('%Y%m%d')

'''Max Price'''
y = data.iloc[:, [11]].values
'''Date and Description'''
X = data.iloc[:, [0,4]].values 

X[:,0] = date_time
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2 , random_state = 42)



X_train = np.array(X_train)
y_train = np.array(y_train)



label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(X_train[:,0])


integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
train_dates = X_train[:,0]
labels = X_train[:,1]
train_dates = train_dates.reshape(len(train_dates), 1 )

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(labels)

encoded_labels = integer_encoded.reshape(len(integer_encoded), 1)




'''A Linear Regression Model: that predicts price from a date input'''
model = LinearRegression()
dependent_variables = np.hstack((train_dates, encoded_labels))
model.fit(dependent_variables,y_train)

filename = 'ecx_linear_regression_model.sav'
pickle.dump(model, open(filename, 'wb'))




from sklearn.metrics import mean_squared_error

loaded_model = pickle.load(open(filename, 'rb'))

result_train = loaded_model.predict(dependent_variables)


rms_training = np.sqrt(mean_squared_error(y_train, result_train))

print rms_training