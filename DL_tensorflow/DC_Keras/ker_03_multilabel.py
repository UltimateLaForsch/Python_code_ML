# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 21:36:18 2021

@author: Ultimate LaForsch
"""

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

import tensorflow.keras as keras
import pandas as pd
import numpy as np


data_path = 'Data\\irrigation_machine.csv'
dataset = pd.read_csv(data_path)

sensors_train = np.array(dataset.iloc[:1500, 1:21])
sensors_test = np.array(dataset.iloc[1500:, 1:21])

parcels_train = np.array(dataset.iloc[:1500, 21:])
parcels_test = np.array(dataset.iloc[1500:, 21:])


model = keras.Sequential()
  
model.add(keras.layers.Dense(units=64, input_shape=(20,), activation='relu'))
model.add(keras.layers.Dense(units=3, activation='sigmoid'))
  
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

# Train for 100 epochs using a validation split of 0.2
model.fit(sensors_train, parcels_train, epochs=100, validation_split=0.2)

# Predict on sensors_test and round up the predictions
preds = model.predict(sensors_test)
preds_rounded = np.round(preds)

# Print rounded preds
print('\nRounded Predictions: \n', preds_rounded)

# Evaluate your model's accuracy on the test data
accuracy = model.evaluate(sensors_test, parcels_test)

# Print accuracy
print('\nAccuracy:', accuracy)
