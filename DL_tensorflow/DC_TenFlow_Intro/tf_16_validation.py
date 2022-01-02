# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 19:54:19 2021

@author: Ultimate LaForsch
"""

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize

data_path = 'Data\\slmnist.csv'

sign_df = pd.read_csv(data_path)

labels = np.array(sign_df.iloc[:, 0], dtype=np.float32)
category_indices = labels
unique_category_count = 4
sign_language_labels = tf.one_hot(category_indices, unique_category_count)

features = np.array(sign_df.iloc[:, 1:], dtype=np.float32)
sign_language_features = normalize(features)


# Define sequential model
model = keras.Sequential()

# Define the first layer
model.add(keras.layers.Dense(32, activation='sigmoid', input_shape=(784,)))

# Add activation function to classifier
model.add(keras.layers.Dense(4, activation='softmax'))

# Set the optimizer, loss function, and metrics
model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Add the number of epochs and the validation split
model.fit(sign_language_features, sign_language_labels, epochs=10, validation_split=0.1)
