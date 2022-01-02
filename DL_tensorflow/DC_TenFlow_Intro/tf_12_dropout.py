# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 16:53:33 2021

@author: Ultimate LaForsch
"""

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

from tensorflow import keras, Variable, matmul, ones, random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


data_path = 'Data\\uci_credit_card.csv'
borrowers = pd.read_csv(data_path)
education = np.array(borrowers['EDUCATION'], dtype=np.float32)
marriage = np.array(borrowers['MARRIAGE'], dtype=np.float32)
age = np.array(borrowers['AGE'], dtype=np.float32)
borrower_features = np.array(borrowers.iloc[:, 1:24], dtype=np.float32)
# borrower_features = borrower_features.T
target = np.array(borrowers['default.payment.next.month'], dtype=np.float32)
default = target.reshape(-1, 1)


# Define the model
def model(w1, b1, w2, b2, features = borrower_features):
 	# Apply relu activation functions to layer 1
 	layer1 = keras.activations.relu(matmul(features, w1) + b1)
    # Apply dropout rate of 0.25
 	dropout = keras.layers.Dropout(rate=0.25)(layer1)
 	return keras.activations.sigmoid(matmul(dropout, w2) + b2)

# Define the loss function
def loss_function(w1, b1, w2, b2, features = borrower_features, targets = default):
 	predictions = model(w1, b1, w2, b2)
 	# Pass targets and predictions to the cross entropy loss
 	return keras.losses.binary_crossentropy(targets, predictions)

def confusion_matrix(default, model_predictions):
	df = pd.DataFrame(np.hstack([default, model_predictions.numpy() > 0.5]), columns = ['Actual','Predicted'])
	confusion_matrix = pd.crosstab(df['Actual'], df['Predicted'], rownames=['Actual'], colnames=['Predicted'])
	sns.heatmap(confusion_matrix, cmap="Greys", fmt="d", annot=True, cbar=False)
	plt.show()

    
test_features = borrower_features    
test_targets = default

# Define the layer 1 weights
w1 = Variable(random.normal([23, 7]), dtype=np.float32)

# Initialize the layer 1 bias
b1 = Variable(ones([7]), dtype=np.float32)

# Define the layer 2 weights
w2 = Variable(random.normal([7, 1]), dtype=np.float32)

# Define the layer 2 bias
b2 = Variable([0], dtype=np.float32)

opt = keras.optimizers.Adam()

# Train the model
for j in range(100):
    # Complete the optimizer
	opt.minimize(lambda: loss_function(w1, b1, w2, b2), var_list=[w1, b1, w2, b2])

# Make predictions with model using test features
model_predictions = model(w1, b1, w2, b2, test_features)

# Construct the confusion matrix
confusion_matrix(test_targets, model_predictions)

