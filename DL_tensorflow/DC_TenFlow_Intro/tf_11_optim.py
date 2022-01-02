# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 22:57:02 2021

@author: Ultimate LaForsch
"""
import numpy as np
# import tensorflow as tf
from tensorflow import divide, Variable, keras
import math

def loss_function(x):
	return 4.0 * math.cos(x-1) + divide(math.cos(2.0 * math.pi * x), x)


# Initialize x_1 and x_2
x_1 = Variable(6.0, dtype=np.float32)
x_2 = Variable(0.3, dtype=np.float32)

# Define the optimization operation
opt = keras.optimizers.SGD(learning_rate=0.01)

for j in range(100):
	# Perform minimization using the loss function and x_1
	opt.minimize(lambda: loss_function(x_1), var_list=[x_1])
	# Perform minimization using the loss function and x_2
	opt.minimize(lambda: loss_function(x_2), var_list=[x_2])

# Print x_1 and x_2 as numpy arrays
print('\n2 different minima, depending on the starting point:')
print('(the right one is only a local minimum)')
print(x_1.numpy(), x_2.numpy())

# used the same optimizer and loss function, but two different
# initial values. When we started at 6.0 with x_1, we found the global
# minimum at 6.02, marked by the dot on the right. When we started at
# 0.3, we stopped around 0.25 with x_2, the local minimum marked by
# a dot on the far left.

# Initialize x_1 and x_2
x_3 = Variable(6.0, np.float32)
x_4 = Variable(0.3, np.float32)

# Define the optimization operation for opt_1 and opt_2
opt_1 = keras.optimizers.SGD(learning_rate=0.01, momentum=0.99)
opt_2 = keras.optimizers.SGD(learning_rate=0.01, momentum=0.00)

for j in range(100):
	opt_1.minimize(lambda: loss_function(x_3), var_list=[x_3])
    # Define the minimization operation for opt_2
	opt_2.minimize(lambda: loss_function(x_4), var_list=[x_4])

# Print x_1 and x_2 as numpy arrays
print('Now with momentum(left) and without (right)')
print(x_3.numpy(), x_4.numpy())
