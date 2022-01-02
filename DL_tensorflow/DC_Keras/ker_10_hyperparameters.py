# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 18:02:02 2021

@author: Ultimate LaForsch
"""

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

import tensorflow.keras as keras
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.datasets import load_breast_cancer
import pandas as pd

# Creates a model given an activation and learning rate
def create_model(learning_rate, activation):
  
  	# Create an Adam optimizer with the given learning rate
  	opt = keras.optimizers.Adam(lr = learning_rate)
  	
  	# Create your binary classification model  
  	model = keras.Sequential()
  	model.add(keras.layers.Dense(128, input_shape = (30,), activation=activation))
  	model.add(keras.layers.Dense(256, activation=activation))
  	model.add(keras.layers.Dense(1, activation = 'sigmoid'))
  	
  	# Compile your model with your optimizer, loss, and metrics
  	model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])
  	return model

  
dataset = load_breast_cancer()
X_pre = dataset.data
X = tf.keras.utils.normalize(X_pre, axis=1)
y = dataset.target


# Create a KerasClassifier
model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model)

# Define the parameters to try out
params = {'activation': ['relu','tanh'], 'batch_size': [32, 128, 256], 
           'epochs': [50, 100, 200], 'learning_rate': [0.1, 0.01, 0.001]}
# params = {'activation': ['relu','tanh'], 'batch_size': [32], 
#           'epochs': [1], 'learning_rate': [0.1]}
# # Create a randomize search cv object passing in the parameters to try
random_search = RandomizedSearchCV(model, param_distributions=params, cv =KFold(3))

random_search.fit(X, y)
result_param = random_search.cv_results_['params']
result_score = pd.DataFrame(random_search.cv_results_['mean_test_score'])
print('\n')
for index, row in result_score.iterrows():
    print(result_param[index])
    print(row)
    

# Create a KerasClassifier
# model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn = create_model(learning_rate=0.01,
#                    activation='relu'), epochs=200, batch_size=32, verbose = 0)

# model.fit(X, y)

# model.score(X, y)

# Calculate the accuracy score for each fold
# kfolds = cross_val_score(model, X, y, cv = 3)

# # Print the mean accuracy
# print('The mean accuracy was:', kfolds.mean())

# # Print the accuracy standard deviation
# print('With a standard deviation of:', kfolds.std())