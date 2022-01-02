# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 17:01:47 2021

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

games_tourney_train = games_tourney.iloc[:3430, ]
games_tourney_test = games_tourney.iloc[3430:, ]

# Define the input
input_tensor = Input((2,))

# Define the output
output_tensor = Dense(2)(input_tensor)

# Create a model
model = Model(input_tensor, output_tensor)

# Compile the model
model.compile(optimizer='adam', loss='mean_absolute_error')
model.fit(games_tourney_train[['seed_diff', 'pred']], games_tourney_train[['score_1', 'score_2']],
          batch_size=16384, epochs=100, verbose=True)
print('\nModel weights:')
print(model.get_weights())
print('\nColumn means:')
print(games_tourney_train.mean())
print('\nEvaluation:')
print(model.evaluate(games_tourney_test[['seed_diff', 'pred']], games_tourney_test[['score_1', 'score_2']],
                     verbose=True))








