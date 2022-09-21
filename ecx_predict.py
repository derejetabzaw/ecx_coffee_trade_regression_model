#!/usr/bin/python
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
import os 
import numpy as np 
from numpy import argmax
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle 
import tools

path = "Coffee_trade_data.csv"
data = pd.read_csv(path)
filename = 'ecx_linear_regression_model.sav'

'''Max Price'''
y = data.iloc[:, [11]].values


'''Date and Description'''
X = data.iloc[:, [0,4]].values 

date_time = (pd.to_datetime(data['Trade Date'])).dt.strftime('%Y%m%d')

X[:,0] = date_time

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2 , random_state = 42)

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))



'''Test Dates'''
test_dates =  X_test[:,0]
labels = X_test[:,1]
test_dates = test_dates.reshape(len(test_dates), 1 )

'''Coffee Description Labels'''
encoded_test_labels = tools.encode(labels)

'''Stacking the Dates with Description'''
test_variable = np.hstack((test_dates, encoded_test_labels))


'''Model Price Prediction'''
result = loaded_model.predict(test_variable)
# print result








'''For instance the price for a specified coffee commodity '''


# random_date = np.array(['20230610','20420810']).astype(float)
# random_date = random_date.reshape(len(random_date), 1)
# specific_labels = ['Unwashed Yiregachefe AQ1','Local Washed  West Arsi 1']

# random_labels = tools.encode_specific_labels(encoded_test_labels,labels,specific_labels)

# variables = np.hstack((random_date,random_labels))

# result = loaded_model.predict(variables)


# print result



from sklearn.metrics import mean_squared_error

rms_test = np.sqrt(mean_squared_error(y_test, result))
print rms_test