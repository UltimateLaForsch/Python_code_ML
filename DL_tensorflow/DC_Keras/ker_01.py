# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 22:10:30 2021

@author: Ultimate LaForsch
"""

# Import the Sequential model and Dense layer
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

import tensorflow.keras as keras
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

    
data_path = 'Data\\banknotes.csv'
banknotes = pd.read_csv(data_path)
# Use pairplot and set the hue to be our class column
sns.pairplot(banknotes, hue='class')

# Show the plot
plt.show()

# Describe the data
print('Dataset stats: \n', banknotes.describe())

# Count the number of observations per class
print('Observations per class: \n', banknotes['class'].value_counts())

X_train = np.array(banknotes.iloc[0:1000, 0:4])
y_train = np.array(banknotes.iloc[0:1000, 4])
X_test = np.array(banknotes.iloc[1000:, 0:4])
y_test = np.array(banknotes.iloc[1000:, 4])


# Create a Sequential model
model = keras.Sequential()

# Add an input layer and a hidden layer with 10 neurons
model.add(keras.layers.Dense(units=1, input_shape=(4,), activation="sigmoid"))

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=20)

accuracy  = model.evaluate(X_test, y_test)[1]
print('\nAccuracy:', accuracy)

