# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 17:26:23 2021

@author: Ultimate LaForsch
"""


from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
import numpy as np


def predict_text(test_text, model=Sequential()):
    if len(test_text.split()) != 3:
        print('Text input should be 3 words!')
        return False
  
    # Turn the test_text into a sequence of numbers
    test_seq = tokenizer.texts_to_sequences([test_text])
    test_seq = np.array(test_seq)
      
    # Use the model passed as a parameter to predict the next word
    pred = model.predict(test_seq).argmax(axis = 1)[0]
      
    # Return the word that maps to the prediction
    return tokenizer.index_word[pred]


text = 'it is not the strength of the body but the strength of the spirit' \
       ' it is useless to meet revenge with revenge it will heal nothing even' \
       ' the smallest person can change the course of history all we have to' \
       ' decide is what to do with the time that is given us the burned hand' \
           ' teaches best after that advice about fire goes to the heart'
vocab_size = 44

# Split text into an array of words 
words = text.split()


# Make sentences of 4 words each, moving one word at a time
sentences = []
seq_len = 4
for i in range(seq_len, len(words)):
  sentences.append(' '.join(words[i - seq_len:i]))

# Instantiate a Tokenizer, then fit it on the sentences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)

# Turn sentences into a sequence of numbers
sequences = tokenizer.texts_to_sequences(sentences)
print("Sentences: \n {} \n Sequences: \n {}".format(sentences[:5],sequences[:5]))

model = Sequential()

# Add an Embedding layer with the right parameters
model.add(Embedding(input_dim=vocab_size, input_length=3, output_dim=8, ))

# Add a 32 unit LSTM layer
model.add(LSTM(32))

# Add a hidden Dense layer of 32 units and an output layer of vocab_size with softmax
model.add(Dense(32, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
model.summary()
model.compile(optimizer='Adam', loss='categorical_crossentropy')

prediction = predict_text('meet revenge with', model)
print('\nPrediction of next word: ', prediction)

