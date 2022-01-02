# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 13:50:08 2021

@author: Ultimate LaForsch
"""

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

from tensorflow import keras, Variable, matmul, ones
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_path = 'Data\\uci_credit_card.csv'
borrowers = pd.read_csv(data_path)
education = np.array(borrowers['EDUCATION'], dtype=np.float32)
marriage = np.array(borrowers['MARRIAGE'], dtype=np.float32)
age = np.array(borrowers['AGE'], dtype=np.float32)
borrower_features = np.array([education, marriage, age])
borrower_features = borrower_features.T
target = np.array(borrowers['default.payment.next.month'], dtype=np.float32)
target = target.reshape(-1, 1)

# Initialize bias1
bias1 = Variable(1.0)

# Initialize weights1 as 3x2 variable of ones
# weights1 = Variable(ones((3, 2)), dtype=np.float32)
weights1 = Variable(np.array([[-0.6, 0.6],
                              [0.8, -0.3],
                              [-0.09, -0.08]]), dtype=np.float32)

# Perform matrix multiplication of borrower_features and weights1
product1 = matmul(borrower_features, weights1)
p = product1.numpy()
# Apply sigmoid activation function to product1 + bias1
dense1 = keras.activations.sigmoid(product1 + bias1)
q = dense1.numpy()
# Print shape of dense1
print("\nDense Layer 1's output shape: {}".format(dense1.shape))

# Initialize bias2 and weights2
bias2 = Variable(1.0)
weights2 = Variable(ones((2, 1)), dtype=np.float32)

# Perform matrix multiplication of dense1 and weights2
product2 = matmul(dense1, weights2)

# Apply activation to product2 + bias2 and print the prediction
prediction = keras.activations.sigmoid(product2 + bias2)
r = prediction.numpy()
print('\nOutput vector\'s shape: ', prediction.shape)
print('\nPrediction of 1st (of 30,000): {}'.format(prediction.numpy()[0, 0]))
print('\nActual value: 1')

loss_vector = prediction - target
loss_vec_np = loss_vector.numpy()
loss = np.sum(loss_vector)
print(loss)
