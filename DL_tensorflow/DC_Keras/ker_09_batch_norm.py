# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 17:38:18 2021

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

def compare_histories_acc(h1,h2):
  plt.plot(h1.history['accuracy'])
  plt.plot(h1.history['val_accuracy'])
  plt.plot(h2.history['accuracy'])
  plt.plot(h2.history['val_accuracy'])
  plt.title("Batch Normalization Effects")
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend(['Train', 'Test', 'Train with Batch Normalization', 'Test with Batch Normalization'], loc='best')
  plt.show()
  
  
data_path = 'Data\\digits_pixels.npy'
digits = np.load(data_path)
labels_pre = np.load('Data\\digits_target.npy')
labels = keras.utils.to_categorical(labels_pre)
X_train = digits[:1257]
X_test = digits[1257:]
y_train = labels[:1257]
y_test = labels[1257:]

# Build your deep network
standard_model = keras.Sequential()
standard_model.add(keras.layers.Dense(50, input_shape=(64,), activation='relu', kernel_initializer='normal'))
standard_model.add(keras.layers.Dense(50, activation='relu', kernel_initializer='normal'))
standard_model.add(keras.layers.Dense(50, activation='relu', kernel_initializer='normal'))
standard_model.add(keras.layers.Dense(10, activation='softmax', kernel_initializer='normal'))

# Compile your model with sgd
standard_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# Build your deep network
batchnorm_model = keras.Sequential()
batchnorm_model.add(keras.layers.Dense(50, input_shape=(64,), activation='relu', kernel_initializer='normal'))
batchnorm_model.add(keras.layers.BatchNormalization())
batchnorm_model.add(keras.layers.Dense(50, activation='relu', kernel_initializer='normal'))
batchnorm_model.add(keras.layers.BatchNormalization())
batchnorm_model.add(keras.layers.Dense(50, activation='relu', kernel_initializer='normal'))
batchnorm_model.add(keras.layers.BatchNormalization())
batchnorm_model.add(keras.layers.Dense(10, activation='softmax', kernel_initializer='normal'))

# Compile your model with sgd
batchnorm_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# Train your standard model, storing its history callback
h1_callback = standard_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, verbose=0)

# Train the batch normalized model you recently built, store its history callback
h2_callback = batchnorm_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, verbose=0)

# Call compare_histories_acc passing in both model histories
compare_histories_acc(h1_callback, h2_callback)