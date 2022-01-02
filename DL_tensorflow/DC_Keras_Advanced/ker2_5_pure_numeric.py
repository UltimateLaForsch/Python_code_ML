# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 22:10:58 2021

@author: Ultimate LaForsch
"""
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import pandas as pd 

games_season = pd.read_csv('Data\\games_season.csv') 
games_tourney = pd.read_csv('Data\\games_tourney_incl_pred.csv')

games_tourney_train = games_tourney.iloc[:3168, ]
games_tourney_test = games_tourney.iloc[3168:, ]


# Create an input layer with 3 columns
input_tensor = Input((3,))

# Pass it to a Dense layer with 1 unit
output_tensor = Dense(1)(input_tensor)

# Create a model
model = Model(input_tensor, output_tensor)

# Compile the model
model.compile(optimizer='adam', loss='mean_absolute_error')


model.fit(games_tourney_train[['home', 'seed_diff', 'pred']],
          games_tourney_train['score_diff'],
          epochs=1,
          verbose=True)

# Evaluate the model on the games_tourney_test dataset
print(model.evaluate(games_tourney_test[['home', 'seed_diff', 'pred']],
          games_tourney_test['score_diff'],
          verbose=True))

