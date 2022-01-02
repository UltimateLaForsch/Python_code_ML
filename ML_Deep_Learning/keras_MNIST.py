import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

df = pd.read_csv('Data/mnist.csv')

raw_y = df.iloc[:, 0]
label_as_binary = LabelBinarizer()
y = label_as_binary.fit_transform(raw_y)

X = np.array(df.iloc[:, 1:])


n_cols = X.shape[1]
model = Sequential()
model.add(Dense(50, activation='relu', input_shape=(n_cols,)))
model.add(Dense(50, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=X, y=y, validation_split=0.3, epochs=100)

