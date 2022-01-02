# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 16:46:55 2021

@author: Ultimate LaForsch
"""

import tensorflow as tf
from tensorflow import GradientTape, Variable, multiply

def compute_gradient(x0):
  	# Define x as a variable with an initial value of x0
	x = Variable(x0)
	with GradientTape() as tape:
		tape.watch(x)
        # Define y using the multiply operation
		y = multiply(x, x)
    # Return the gradient of y with respect to x
	return tape.gradient(y, x).numpy()

# Compute and print gradients at x = -1, 1, and 0
print('Gradients:')
print(compute_gradient(-1.0))
print(compute_gradient(1.0))
print(compute_gradient(0.0))
