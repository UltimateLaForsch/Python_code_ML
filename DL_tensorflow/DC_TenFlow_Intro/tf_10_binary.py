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
bill_amounts = np.array(borrowers['BILL_AMT1'], dtype=np.float32)
bill_amounts = (bill_amounts.T).reshape(-1, 1)
age = np.array(borrowers['AGE'], dtype=np.float32)
borrower_features = np.array([education, marriage, age])
borrower_features = borrower_features.T
target = np.array(borrowers['default.payment.next.month'], dtype=np.float32)
target = target.reshape(-1, 1)

# Construct input layer from features
inputs = tf.constant(bill_amounts, dtype=float)

# Define first dense layer
dense1 = keras.layers.Dense(3, activation='relu')(inputs)

# Define second dense layer
dense2 = keras.layers.Dense(2, activation='relu')(dense1)

# Define output layer
outputs = keras.layers.Dense(1, activation='sigmoid')(dense2)

# Print error for first five examples
error = target[:5] - outputs.numpy()[:5]
print(error)
