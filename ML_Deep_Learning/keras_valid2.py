import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

df = pd.read_csv('Data/titanic_all_numeric.csv')
# print(dataset.describe())
predictors = np.array(df.iloc[:, 1:11])
predictors = predictors.astype(np.float32)
n_cols = predictors.shape[1]
target = np.array(df.iloc[:, 0])
target = to_categorical(target)

n_cols = predictors.shape[1]
model_1 = Sequential()
model_1.add(Dense(10, activation='relu', input_shape=(n_cols,)))
model_1.add(Dense(10, activation='relu'))
model_1.add(Dense(units=2, activation='softmax'))
model_1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
early_stopping_monitor = EarlyStopping(patience=2)


model_2 = Sequential()
model_2.add(Dense(100, activation='relu', input_shape=(n_cols,)))
model_2.add(Dense(100, activation='relu'))
model_2.add(Dense(units=2, activation='softmax'))
model_2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model_1_training = model_1.fit(x=predictors, y=target, validation_split=0.2, epochs=15, callbacks=early_stopping_monitor)
model_2_training = model_2.fit(x=predictors, y=target, validation_split=0.2, epochs=15, callbacks=early_stopping_monitor)

# Create the plot
plt.plot(model_1_training.history['val_loss'], 'r', model_2_training.history['val_loss'], 'b')
plt.xlabel('Epochs')
plt.ylabel('Validation score')
plt.show()