# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 22:22:26 2021

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

def plot_loss(loss,val_loss):
  plt.figure()
  plt.plot(loss)
  plt.plot(val_loss)
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper right')
  plt.show()


def plot_accuracy(acc,val_acc):
  # Plot training & validation accuracy values
  plt.figure()
  plt.plot(acc)
  plt.plot(val_acc)
  plt.title('Model accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper left')
  plt.show()

  
data_path = 'Data\\irrigation_machine.csv'
dataset = pd.read_csv(data_path)

X_train = np.array(dataset.iloc[:1500, 1:21])
X_test = np.array(dataset.iloc[1500:, 1:21])

y_train = np.array(dataset.iloc[:1500, 21:])
y_test = np.array(dataset.iloc[1500:, 21:])


model = keras.Sequential()
  
model.add(keras.layers.Dense(units=64, input_shape=(20,), activation='relu'))
model.add(keras.layers.Dense(units=3, activation='sigmoid'))
  
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

h_callback = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# Plot train vs test loss during training
plot_loss(h_callback.history['loss'], h_callback.history['val_loss'])

# # Plot train vs test accuracy during training
plot_accuracy(h_callback.history['accuracy'], h_callback.history['val_accuracy'])
