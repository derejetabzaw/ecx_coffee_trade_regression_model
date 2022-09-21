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
import pickle 



path = 'Last Three years Coffee trade data.csv'
data = pd.read_csv(path)

X = data.iloc[:, [11,14,17]].values
y = data.iloc[:, [0,4]].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2 , random_state = 42)



X_train = np.array(X_train)
y_train = np.array(y_train)


# print X_train,y_train
print y_train[0][1]

# label_encoder = LabelEncoder()
# integer_encoded = label_encoder.fit_transform(y_train)


# integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
# model = LinearRegression(random_state=0).fit(X_train,integer_encoded)



# filename = 'finalized_model.sav'
# pickle.dump(model, open(filename, 'wb'))

