#!/usr/bin/python
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
import numpy as np 
from sklearn.preprocessing import LabelEncoder



def encode(input_data):

	label_encoder = LabelEncoder()
	integer_encoded = label_encoder.fit_transform(input_data)

	encoded_data = integer_encoded.reshape(len(integer_encoded), 1)


	return encoded_data



def decoder(input_data,encoded_data):

	label_encoder = LabelEncoder()
	integer_encoded = label_encoder.fit_transform(input_data)
	
	decoded_data = label_encoder.inverse_transform(encoded_data)

	decoded_data = decoded_data.reshape(len(decoded_data), 1)

	return decoded_data


def encode_specific_labels(encoded_labels,labels,specific_label):
	random_labels = []
	for i in range(len(specific_label)):
		random_labels.append(encoded_labels[np.where(labels ==specific_label[i])][0][0])


	random_labels = np.array(random_labels)
	random_labels = random_labels.reshape(len(random_labels), 1)

	return random_labels