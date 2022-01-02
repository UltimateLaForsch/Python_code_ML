# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 22:04:10 2021

@author: Ultimate LaForsch
"""
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
    
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import pandas as pd 

games_tourney = pd.read_csv('Data\\games_tourney.csv') 
games_tourney.head() 
games_tourney_train = games_tourney.iloc[:3430,]
X_test = games_tourney.iloc[3430:, 4]
y_test = games_tourney.iloc[3430:, 5]


input_tensor = Input(shape=(1,))

# output in two steps...
# output_layer = Dense(1)
# output_tensor = output_layer(input_tensor)

# ... and in one step
output_tensor = Dense(1)(input_tensor)

model = Model(input_tensor, output_tensor)
model.compile(optimizer='adam', loss='mae')
model.summary()

plot_model(model, to_file='Data\\model.png')
img = plt.imread('Data\\model.png')
plt.imshow(img)
plt.show()

model.fit(games_tourney_train['seed_diff'], games_tourney_train['score_diff'],
          epochs=1,
          batch_size=128,
          validation_split=0.1,
          verbose=True)
print(model.evaluate(X_test, y_test))


