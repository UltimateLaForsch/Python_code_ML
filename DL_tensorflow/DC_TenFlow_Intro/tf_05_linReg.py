# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 10:20:24 2021

@author: Ultimate LaForsch
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_results(intercept, slope):
	size_range = np.linspace(6,14,100)
	price_pred = [intercept+slope*s for s in size_range]
	plt.scatter(size_log, price_log, color = 'black', s=1)
	plt.plot(size_range, price_pred, linewidth=3.0, color='red')
	plt.xlabel('log(size)')
	plt.ylabel('log(price)')
	plt.title('Scatterplot of data and fitted regression line')
	plt.show()
    
    
data_path = 'Data\kc_house_data.csv'
housing = pd.read_csv(data_path)

size = np.array(housing['sqft_lot'], dtype=np.float32)
price = np.array(housing['price'], dtype=np.float32)
size_log = np.log(size)
price_log = np.log(price)

# Define a linear regression model
def linear_regression(intercept, slope, features = size_log):
	return slope * features + intercept

# Set loss_function() to take the variables as arguments
def loss_function(intercept, slope, features = size_log, targets = price_log):
	# Set the predicted values
	predictions = linear_regression(intercept, slope, features)
    
    # Return the mean squared error loss
	return keras.losses.mse(targets, predictions)



# Compute the loss for different slope and intercept values
print('2 test loss function values:')
print(loss_function(0.1, 0.1).numpy())
print(loss_function(0.1, 0.5).numpy())

# Train a linear model
intercept = tf.Variable(5., np.float32)
slope = tf.Variable(0.001, np.float32)

# Initialize an Adam optimizer
opt = keras.optimizers.Adam(0.5)

for j in range(100):
	# Apply minimize, pass the loss function, and supply the variables
	opt.minimize(lambda: loss_function(intercept, slope), var_list=[intercept, slope])

	# Print every 10th value of the loss
	if j % 10 == 0:
		print(loss_function(intercept, slope).numpy())

# Plot data and regression line
plot_results(intercept, slope)



