# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 18:52:07 2021

@author: Ultimate LaForsch
"""

import tensorflow as tf
from tensorflow import keras



# Define a Keras sequential model
model = keras.Sequential()

# Define the first dense layer
model.add(keras.layers.Dense(16, activation='relu', input_shape=(784,)))

# Define the second dense layer
model.add(keras.layers.Dense(8, activation='relu'))

# Define the output layer
model.add(keras.layers.Dense(4, activation='softmax'))

# Print the model architecture
print(model.summary())

model2 = keras.Sequential()
# Define the first dense layer
model2.add(keras.layers.Dense(16, activation='sigmoid', input_shape=(784,)))

# Apply dropout to the first layer's output
model2.add(keras.layers.Dropout(0.25))

# Define the output layer
model2.add(keras.layers.Dense(4, activation='softmax'))

# Compile the model
model2.compile('adam', loss='categorical_crossentropy')

# Print a model summary
print(model2.summary())