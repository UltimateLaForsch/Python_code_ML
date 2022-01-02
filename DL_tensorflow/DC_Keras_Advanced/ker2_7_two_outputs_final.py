
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import pandas as pd
from scipy.special import expit as sigmoid

games_season = pd.read_csv('Data\\games_season.csv')
games_tourney = pd.read_csv('Data\\games_tourney_incl_pred.csv')

games_tourney_train = games_tourney.iloc[:3430, ]
games_tourney_test = games_tourney.iloc[3430:, ]

# Create an input layer with 2 columns
input_tensor = Input(shape=(2,))

# Create the first output
output_tensor_1 = Dense(1, activation='linear', use_bias=False)(input_tensor)

# Create the second output (use the first output as input here)
output_tensor_2 = Dense(1, activation='sigmoid', use_bias=False)(output_tensor_1)

# Create a model with 2 outputs
model = Model(input_tensor, [output_tensor_1, output_tensor_2])

model.compile(loss=['mean_absolute_error', 'binary_crossentropy'], optimizer=Adam(learning_rate=0.01))
model.fit(x=games_tourney_train[['seed_diff', 'pred']],
          y=[games_tourney_train[['score_diff']], games_tourney_train[['won']]],
          batch_size=16384,
          epochs=10,
          verbose=True)

print('\nModel weights:')
print(model.get_weights())
print('\nColumn means:')
print(games_tourney_train.mean())

# Weight from the model
weight = 0.14

# Print the approximate win probability predicted close game
print(sigmoid(1 * weight))

# Print the approximate win probability predicted blowout game
print(sigmoid(10 * weight))

# Evaluate the model on new data
print(model.evaluate(x=games_tourney_test[['seed_diff', 'pred']],
                     y=[games_tourney_test[['score_diff']], games_tourney_test[['won']]],
                     verbose=False))
