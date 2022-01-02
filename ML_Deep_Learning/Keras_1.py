import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


df = pd.read_csv('Data/hourly_wages.csv')
wage_per_hour = np.array(df.iloc[:, 0])
target = wage_per_hour
predictors = np.array(df.iloc[:, 1:10])

feature_no = predictors.shape[1]
# Set up the model: model
model = Sequential()
# Add the first layer
model.add(Dense(50, activation='relu', input_shape=(feature_no,)))
# Add the second layer
model.add(Dense(32, activation='relu'))
# Add the output layer
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x=predictors, y=wage_per_hour, epochs=20, batch_size=32)


# wage_per_hour = np.asmatrix(df.iloc[:, 0])
# wage_per_hour = wage_per_hour.reshape(-1, 1)
# wage_per_hour = wage_per_hour.astype(np.float32)
# wage_per_hour = wage_per_hour.reshape(-1, 1)
# predictors = predictors.astype(np.float32)
# predictors = predictors.reshape(-1, 1)