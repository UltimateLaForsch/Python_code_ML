# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 21:54:50 2021

@author: Ultimate LaForsch
"""

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

from tensorflow.keras.layers import Embedding, Flatten, Input, Subtract, Concatenate, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
# from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
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

# Create an Input for each team
team_in_1 = Input(shape=(1,), name='Team-1-In')
team_in_2 = Input(shape=(1,), name='Team-2-In')

# Create an input for home vs away
home_in = Input(shape=(1,), name='Home-In')

# Lookup the team inputs in the team strength model
team_1_strength = team_strength_model(team_in_1)
team_2_strength = team_strength_model(team_in_2)

# Combine the team strengths with the home input using a Concatenate layer
out = Concatenate()([team_1_strength, team_2_strength, home_in])
# then add a Dense layer
out = Dense(1)(out)

model = Model([team_in_1, team_in_2, home_in], out)
model.compile(optimizer='adam', loss='mean_absolute_error')
model.fit(x=[games_season['team_1'], games_season['team_2'], games_season['home']],
          y=games_season['score_diff'],
          epochs=5,
          batch_size=2048,
          validation_split=0.1,
          verbose=True)
print('\nLoss on Validation Set:')

games_tourney['pred'] = model.predict([games_tourney['team_1'], games_tourney['team_2'],
                                       games_tourney['home']])
games_tourney.to_csv('Data\\games_tourney_incl_pred.csv')


