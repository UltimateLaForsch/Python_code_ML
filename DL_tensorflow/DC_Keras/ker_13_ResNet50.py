# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 16:39:13 2021

@author: Ultimate LaForsch
"""

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50, decode_predictions
import numpy as np

# Load the image with the right target size for your model
img = image.load_img('Data\\dog.jpg', target_size=(224, 224))
# img = image.load_img('Data\\motorcyc.jpg', target_size=(224, 224))

# Turn it into an array
img_array = image.img_to_array(img)

# Expand the dimensions of the image, this is so that it fits the expected model input format
img_expanded = np.expand_dims(img_array, axis = 0)

# Pre-process the img in the same way original images were
img_ready = preprocess_input(img_expanded)
# image_re = image.array_to_img(img_ready)
# image_re.save('tf dog.jpg')

# Instantiate a ResNet50 model with 'imagenet' weights
model = ResNet50(weights='imagenet') 

# Predict with ResNet50 on your already processed img
preds = model.predict(img_ready)

# Decode the first 3 predictions
print('Predicted:', decode_predictions(preds, top=3)[0])


