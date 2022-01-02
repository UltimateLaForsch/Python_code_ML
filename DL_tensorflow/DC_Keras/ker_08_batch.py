# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 16:49:12 2021

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

def get_model():
  model = keras.Sequential()
  model.add(keras.layers.Dense(units=64, input_shape=(20,), activation='relu'))
  model.add(keras.layers.Dense(units=3, activation='sigmoid'))
  model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
  return model


data_path = 'Data\\irrigation_machine.csv'
dataset = pd.read_csv(data_path)

X_train = np.array(dataset.iloc[:300, 1:21])
X_test = np.array(dataset.iloc[300:400, 1:21])

y_train = np.array(dataset.iloc[:300, 21:])
y_test = np.array(dataset.iloc[300:400, 21:])

# Get a fresh new model with get_model
model = get_model()

# Train your model for 5 epochs with a batch size of 1
print('Batch size: 1\n')
model.fit(X_train, y_train, epochs=5, batch_size=1)
print("\n The accuracy when using a batch of size 1 is: ",
      model.evaluate(X_test, y_test)[1])

print('\nBatch size: 300 (whole training set)')
model.fit(X_train, y_train, epochs=5, batch_size=300)
print("\n The accuracy when using the whole training set as batch-size was: ",
      model.evaluate(X_test, y_test)[1])

