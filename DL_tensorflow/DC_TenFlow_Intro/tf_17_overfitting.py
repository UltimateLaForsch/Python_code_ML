# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 20:25:19 2021

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

# only the first 26 examples. only a small subset of the examples
# from the original sign language letters dataset.
# A small sample, coupled with a heavily-parameterized model, will
# generally lead to overfitting. This means that your model will
# simply memorize the class of each example, rather than identifying
# features that generalize to many examples.

labels = np.array(sign_df.iloc[0:25, 0], dtype=np.float32)
category_indices = labels
unique_category_count = 4
sign_language_labels = tf.one_hot(category_indices, unique_category_count)

features = np.array(sign_df.iloc[0:25, 1:], dtype=np.float32)
sign_language_features = normalize(features)

# Define sequential model
model = keras.Sequential()

# Define the first layer
model.add(keras.layers.Dense(1024, activation='relu', input_shape=(784,)))

# Add activation function to classifier
model.add(keras.layers.Dense(4, activation='softmax'))

# Finish the model compilation
model.compile(optimizer=keras.optimizers.Adam(lr=0.001), 
              loss='categorical_crossentropy', metrics=['accuracy'])

# Complete the model fit operation
model.fit(sign_language_features, sign_language_labels, epochs=50, validation_split=0.5)

# Evaluate the small model using the train data
# small_train = model.evaluate(sign_language_features, sign_language_labels)

# # Evaluate the small model using the test data
# small_test = ____

# # Print losses
# print('\n Small - Train: {}, Test: {}'.format(small_train, small_test))