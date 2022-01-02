# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 19:23:03 2021

@author: Ultimate LaForsch
"""

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
from tensorflow import keras
import pandas as pd
import numpy as np

data_path = 'Data\\slmnist.csv'

sign_df = pd.read_csv(data_path)

labels = np.array(sign_df.iloc[0:1000, 0], dtype=np.float32)
category_indices = labels
unique_category_count = 4
sign_language_labels = tf.one_hot(category_indices, unique_category_count)

sign_language_features = np.array(sign_df.iloc[0:1000, 1:], dtype=np.float32)


# Define a sequential model
model = keras.Sequential() 

# Define a hidden layer
model.add(keras.layers.Dense(16, activation='relu', input_shape=(784,)))

# Define the output layer
model.add(keras.layers.Dense(4, activation='softmax'))

# Compile the model
model.compile('SGD', loss='categorical_crossentropy')

# Complete the fitting operation
model.fit(sign_language_features, sign_language_labels, epochs=5)