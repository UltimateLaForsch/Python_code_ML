# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 19:05:01 2021

@author: Ultimate LaForsch
"""

import tensorflow as tf
from tensorflow import keras

pre_model1 = keras.Sequential()
pre_model2 = keras.Sequential()

m1_inputs = keras.Input(shape=(784,))
m2_inputs = keras.Input(shape=(784,))

# For model 1, pass the input layer to layer 1 and layer 1 to layer 2
m1_layer1 = keras.layers.Dense(12, activation='sigmoid')(m1_inputs)
m1_layer2 = keras.layers.Dense(4, activation='softmax')(m1_layer1)

# For model 2, pass the input layer to layer 1 and layer 1 to layer 2
m2_layer1 = keras.layers.Dense(12, activation='relu')(m2_inputs)
m2_layer2 = keras.layers.Dense(4, activation='softmax')(m2_layer1)

# Merge model outputs and define a functional model
merged = keras.layers.add([m1_layer2, m2_layer2])
model = keras.Model(inputs=[m1_inputs, m2_inputs], outputs=merged)

# Print a model summary
print(model.summary())

