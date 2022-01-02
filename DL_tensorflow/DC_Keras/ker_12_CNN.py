# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 17:49:30 2021

@author: Ultimate LaForsch
"""

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras


data_path = 'Data\\X_test_MNIST.npy'
X_pre = np.load(data_path)
# X_test = np.expand_dims(X_pre, axis=-1)
 
X_test = X_pre.reshape(100, 28, 28, -1)
y_test = np.load('Data\\y_test_MNIST.npy')
# y_test = keras.utils.to_categorical(y_pre)
X_test_noise= np.load('Data\\X_test_MNIST_noise.npy')


# Import the Conv2D and Flatten layers and instantiate model
model = keras.Sequential()

# Add a convolutional layer of 32 filters of size 3x3
model.add(keras.layers.Conv2D(32, kernel_size=3, input_shape=(28, 28, 1), activation='relu'))

# Add a convolutional layer of 16 filters of size 3x3
model.add(keras.layers.Conv2D(16, kernel_size=3, activation='relu'))

# Flatten the previous layer output
model.add(keras.layers.Flatten())

# Add as many outputs as classes with softmax activation
model.add(keras.layers.Dense(10, activation = 'softmax'))

model.summary()

# 'categorical_crossentropy' works on one-hot encoded target, while
# 'sparse_categorical_crossentropy' works on integer target.
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(X_test, y_test, epochs=100)


# Obtain a reference to the outputs of the first layer
first_layer_output = model.layers[0].output

# Build a model using the model's input and the first layer output
first_layer_model = keras.Model(inputs = model.layers[0].input, outputs=first_layer_output)

# Use this model to predict on X_test
activations = first_layer_model.predict(X_test)

fig, axs = plt.subplots(1, 2)
axs = axs.flatten()
# Plot the activations of first digit of X_test for the 15th filter
axs[0].matshow(activations[0,:,:,14], cmap = 'viridis')
# Do the same but for the 18th filter now
axs[1].matshow(activations[0,:,:,17], cmap = 'viridis')
plt.title('15th filter (left) vs 18th filter (right)')
plt.show()