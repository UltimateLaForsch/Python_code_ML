# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 22:52:00 2021

@author: Ultimate LaForsch
"""

import tensorflow as tf
from tensorflow import keras, feature_column, estimator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_path = 'Data\kc_house_data.csv'
housing = pd.read_csv(data_path)

bedrooms = feature_column.numeric_column("bedrooms")
bathrooms = feature_column.numeric_column("bathrooms")
labels = feature_column.numeric_column("price")

# Define the list of feature columns
feature_list = [bedrooms, bathrooms]

def input_fn():
	# Define the labels
	labels = np.array(housing['price'])
	# Define the features
	features = {'bedrooms':np.array(housing['bedrooms']), 
                'bathrooms':np.array(housing['bathrooms'])}
	return features, labels

# Define the model and set the number of steps
model = estimator.DNNRegressor(feature_columns=feature_list, hidden_units=[2, 2])
model.train(input_fn, steps=1)

model = estimator.LinearRegressor(feature_columns=feature_list)
model.train(input_fn, steps=2)
