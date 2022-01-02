import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

df = pd.read_csv('Data/titanic_all_numeric.csv')
# print(dataset.describe())
predictors = np.array(df.iloc[:, 1:11])
predictors = predictors.astype(np.float32)
n_cols = predictors.shape[1]
target = np.array(df.iloc[:, 0])
target = to_categorical(target)

n_cols = predictors.shape[1]
model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(n_cols, )))
model.add(Dense(100, activation='relu'))
model.add(Dense(units=2, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
early_stopping_monitor = EarlyStopping(patience=2)
model.fit(x=predictors, y=target, validation_split=0.3, epochs=30, callbacks=early_stopping_monitor)


