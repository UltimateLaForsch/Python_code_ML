# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 17:33:57 2021

@author: Ultimate LaForsch
"""

# import tensorflow as tf
# physical_devices = tf.config.list_physical_devices('GPU')
# for device in physical_devices:
#     tf.config.experimental.set_memory_growth(device, True)

from tensorflow.keras.layers import Embedding, Flatten, Input, Subtract
from tensorflow.keras.models import Model
# from tensorflow.keras.utils import plot_model
# import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np

games_season = pd.read_csv('Data\\games_season.csv') 
print(games_season.head())
games_tourney = pd.read_csv('Data\\games_tourney.csv') 

# Count the unique number of teams
n_teams = np.unique(games_season['team_1']).shape[0]

# Create an embedding layer
team_lookup = Embedding(input_dim=n_teams,
                        output_dim=1,
                        input_length=1,
                        name='Team-Strength')
 
# Create an input layer for the team ID
teamid_in = Input(shape=(1,))

# Lookup the input in the team strength embedding layer
strength_lookup = team_lookup(teamid_in)

# Flatten the output
strength_lookup_flat = Flatten()(strength_lookup)

# Combine the operations into a single, re-usable model
team_strength_model = Model(teamid_in, strength_lookup_flat, name='Team-Strength-Model')

# Input layer for team 1
team_in_1 = Input(shape=(1,), name='Team-1-In')

# Separate input layer for team 2
team_in_2 = Input(shape=(1,), name='Team-2-In')

# Lookup team 1 in the team strength model
team_1_strength = team_strength_model(team_in_1)

# Lookup team 2 in the team strength model
team_2_strength = team_strength_model(team_in_2)
# Subtraction layer
score_diff = Subtract()([team_1_strength, team_2_strength])

model = Model([team_in_1, team_in_2], score_diff)
model.compile(optimizer='adam', loss='mean_absolute_error')

# Get the team_1 column from the regular season data
input_train_1 = games_season['team_1']

# Get the team_2 column from the regular season data
input_train_2 = games_season['team_2']

# Fit the model to input 1 and 2, using score diff as a target
model.fit([input_train_1, input_train_2],
          games_season['score_diff'],
          epochs=1,
          batch_size=2048,
          validation_split=0.1,
          verbose=True)

input_1 = games_tourney['team_1']

# Get the team_2 column from the regular season data
input_2 = games_tourney['team_2']

print('\nLoss on Validation Set:')
loss_val = model.evaluate([input_1, input_2], games_tourney['score_diff'], verbose=False)
print(round(loss_val, 3))

