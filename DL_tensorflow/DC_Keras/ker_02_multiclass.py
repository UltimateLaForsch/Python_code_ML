# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 18:38:33 2021

@author: Ultimate LaForsch
"""

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

import tensorflow.keras as keras
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

    
data_path = 'Data\\darts.csv'
darts = pd.read_csv(data_path)
# Use pairplot and set the hue to be our class column
sns.pairplot(darts, hue='competitor')

# Show the plot
plt.show()

# Transform into a categorical variable
darts.competitor = pd.Categorical(darts.competitor)

# Assign a number to each category (label encoding)
darts.competitor = darts.competitor.cat.codes 

# Print the label encoded competitors
print('\nLabel encoded competitors: \n',darts.competitor.head())

coordinates = darts.drop(['competitor'], axis=1)
X_train = np.array(coordinates.iloc[0:600, 0:4])
X_test = np.array(coordinates.iloc[600:, 0:4])

# Use to_categorical on your labels
competitors = keras.utils.to_categorical(darts.competitor)
y_train = competitors[:600]
y_test = competitors[600:]

# Now print the one-hot encoded labels
print('\nOne-hot encoded competitors: \n',competitors)

# Instantiate a sequential model
model = keras.Sequential()
  
# Add 3 dense layers of 128, 64 and 32 neurons each
model.add(keras.layers.Dense(units=128, input_shape=(2,), activation='relu'))
model.add(keras.layers.Dense(units=64, activation='relu'))
model.add(keras.layers.Dense(units=32, activation='relu'))
  
# Add a dense layer with as many neurons as competitors
model.add(keras.layers.Dense(units=4, activation='softmax'))
  
# Compile your model using categorical_crossentropy loss
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=200)
accuracy = model.evaluate(X_test, y_test)[1]
print('\nAccuracy: ', accuracy)


# Small test set
coords_small_test = np.array([[0.209048, -0.077398],
                              [0.082103, -0.721407],
                              [0.198165, -0.674646],
                              [-0.348660,  0.035086],
                              [0.214726,  0.183894]])
competitors_small_test = np.array([[0., 0., 1., 0.],
       [0., 0., 0., 1.],
       [0., 0., 0., 1.],
       [1., 0., 0., 0.],
       [0., 0., 1., 0.]])
                                  

# Predict on coords_small_test
preds = model.predict(coords_small_test)

# Print preds vs true values
print("{:45} | {}".format('\nRaw Model Predictions','True labels'))
for i,pred in enumerate(preds):
  print("{} | {}".format(pred,competitors_small_test[i]))

# Extract the position of highest probability from each pred vector
preds_chosen = [np.argmax(pred) for pred in preds]

# Print preds vs true values
print("{:10} | {}".format('\nRounded Model Predictions','True labels'))
for i,pred in enumerate(preds_chosen):
  print("{:25} | {}".format(pred,competitors_small_test[i]))