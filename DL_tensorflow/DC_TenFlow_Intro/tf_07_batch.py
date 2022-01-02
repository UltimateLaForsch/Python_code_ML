# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 11:18:55 2021

@author: Ultimate LaForsch
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_path = 'Data\kc_house_data.csv'
housing = pd.read_csv(data_path)

size = np.array(housing['sqft_lot'], dtype=np.float32)
price = np.array(housing['price'], dtype=np.float32)
bedrooms = np.array(housing['bedrooms'], dtype=np.float32)
size_log = np.log(size)
price_log = np.log(price)


# Define the linear regression model
def linear_regression(params, feature1 = size_log, feature2 = bedrooms):
	return params[0] + feature1 * params[1] + feature2 * params[2]


# Define the loss function
def loss_function(params, targets = price_log, feature1 = size_log, feature2 = bedrooms):
	# Set the predicted values
	predictions = linear_regression(params, feature1, feature2)
  
	# Use the mean absolute error loss
	return keras.losses.mae(targets, predictions)

def print_results(params):
	return print('loss: {:0.3f}, intercept: {:0.3f}, slope_1: {:0.3f}, slope_2: {:0.3f}'.format(loss_function(params).numpy(), params[0].numpy(), params[1].numpy(), params[2].numpy()))


# --- Train a linear model

# parameter start values
intercept = tf.Variable(5., np.float32)
slope1 = tf.Variable(0.001, np.float32)
slope2 = tf.Variable(0.001, np.float32)

params = [intercept, slope1, slope2]

# Define the optimize operation
opt = keras.optimizers.Adam()

# Perform minimization and print trainable variables
for j in range(10):
	opt.minimize(lambda: loss_function(params), var_list=[params])
	print_results(params)
    
# now with batches

# parameter start values reset
intercept = tf.Variable(5., np.float32)
slope1 = tf.Variable(0.001, np.float32)
slope2 = tf.Variable(0.001, np.float32)
params = [intercept, slope1, slope2]
print('Batch Training:')
for batch in pd.read_csv(data_path, chunksize=100):
    size_batch = np.array(batch['sqft_lot'], dtype=np.float32)
    price_batch = np.array(batch['price'], dtype=np.float32)
    bedrooms_batch = np.array(batch['bedrooms'], dtype=np.float32)
    size_log_batch = np.log(size)
    price_log_batch = np.log(price)
    opt.minimize(lambda: loss_function(params), var_list=[params])
    print_results(params)


    
    
    