# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 16:08:46 2021

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

def plot_loss(loss,val_loss):
  plt.figure()
  plt.plot(loss)
  plt.plot(val_loss)
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper right')
  plt.show()


def plot_results(train_accs, test_accs):
  plt.plot(training_sizes, train_accs, 'o-', label="Training Accuracy")
  plt.plot(training_sizes, test_accs, 'o-', label="Test Accuracy")
  plt.title('Accuracy vs Number of training samples')
  plt.xlabel('# of training samples')
  plt.ylabel('Accuracy')
  plt.legend(loc="best")
  plt.show()


  
data_path = 'Data\\digits_pixels.npy'
digits = np.load(data_path)
labels_pre = np.load('Data\\digits_target.npy')
labels = keras.utils.to_categorical(labels_pre)
X_train = digits[:1257]
X_test = digits[1257:]
y_train = labels[:1257]
y_test = labels[1257:]

# Instantiate a Sequential model
model = keras.models.Sequential()

# Input and hidden layer with input_shape, 16 neurons, and relu 
model.add(keras.layers.Dense(units=16, input_shape = (64,), activation='relu'))

# Output layer with 10 neurons (one per digit) and softmax
model.add(keras.layers.Dense(units=10, activation='softmax'))

# Compile your model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
initial_weights = model.get_weights()

# Test if your model is well assembled by predicting before training
print(model.predict(X_train))

# Train your model for 60 epochs, using X_test and y_test as validation data
h_callback = model.fit(X_train, y_train, epochs=60, validation_data=(X_test, y_test), verbose=0)

# Extract from the h_callback object loss and val_loss to plot the learning curve
plot_loss(h_callback.history['loss'], h_callback.history['val_loss'])

training_sizes = np.array([ 125,  502,  879, 1255])
train_accs = []
test_accs = []
early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)

for size in training_sizes:
  	# Get a fraction of training data (we only care about the training data)
    X_train_frac, y_train_frac = X_train[:size], y_train[:size]

    # Reset the model to the initial weights and train it on the new training data fraction
    model.set_weights(initial_weights)
    model.fit(X_train_frac, y_train_frac, epochs = 50, callbacks = [early_stop])

    # Evaluate and store both: the training data fraction and the complete test set results
    train_accs.append(model.evaluate(X_train_frac, y_train_frac)[1])
    test_accs.append(model.evaluate(X_test, y_test)[1])
    
# Plot train vs test accuracies
plot_results(train_accs, test_accs)


      