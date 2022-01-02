import numpy as np

def length(x):
    """Compute the length of a vector"""
    length_x = np.sqrt(x.T.dot(x))
    return length_x

x = np.array([1, 0])
print(length(x))
