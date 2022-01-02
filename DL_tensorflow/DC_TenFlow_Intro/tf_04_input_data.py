# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 22:25:20 2021

@author: Ultimate LaForsch
"""
import numpy as np
import pandas as pd
import tensorflow as tf

data_path = 'Data\kc_house_data.csv'
housing = pd.read_csv(data_path)
print(housing['price'])

price = np.array(housing['price'], np.float32)
waterfront = tf.cast(housing['waterfront'], tf.bool)
print(price)
print(waterfront)

from tensorflow import keras
