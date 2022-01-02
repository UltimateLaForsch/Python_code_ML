# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 17:21:39 2021

@author: Ultimate LaForsch
"""

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

import tensorflow.keras as keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_model(act_function):
  if act_function not in ['relu', 'leaky_relu', 'sigmoid', 'tanh']:
    raise ValueError('Make sure your activation functions are named correctly!')
  print("Working with ",act_function,"...")
  model = keras.Sequential()
  model.add(keras.layers.Dense(units=64, input_shape=(20,), activation=act_function))
  model.add(keras.layers.Dense(units=3, activation='sigmoid'))
  model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
  return model


data_path = 'Data\\irrigation_machine.csv'
dataset = pd.read_csv(data_path)

X_train = np.array(dataset.iloc[:1500, 1:21])
X_test = np.array(dataset.iloc[1500:, 1:21])

y_train = np.array(dataset.iloc[:1500, 21:])
y_test = np.array(dataset.iloc[1500:, 21:])

activations = ['relu', 'sigmoid', 'tanh']

# Loop over the activation functions
activation_results = {}
val_loss_per_function = {}
val_acc_per_function = {}
for act in activations:
  # Get a new model with the current activation
  model = get_model(act)
  # Fit the model and store the history results
  h_callback = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, verbose=0)
  activation_results[act] = h_callback 
  val_loss_per_function[act] = h_callback.history['val_loss']
  val_acc_per_function[act] = h_callback.history['val_accuracy']

# Create a dataframe from val_loss_per_function
val_loss= pd.DataFrame(val_loss_per_function)

# Call plot on the dataframe
val_loss.plot()
plt.show()

# Create a dataframe from val_acc_per_function
val_acc = pd.DataFrame(val_acc_per_function)

# Call plot on the dataframe
val_acc.plot()
plt.show()
