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

# Define the first dense layer
dense1 = keras.layers.Dense(7, activation='sigmoid')(borrower_features)

# Define a dense layer with 3 output nodes
dense2 = keras.layers.Dense(3, activation='sigmoid')(dense1)

# Define a dense layer with 1 output node
predictions = keras.layers.Dense(1, activation='sigmoid')(dense2)

# Print the shapes of dense1, dense2, and predictions
print('\n shape of dense1: ', dense1.shape)
print('\n shape of dense2: ', dense2.shape)
print('\n shape of predictions: ', predictions.shape)
