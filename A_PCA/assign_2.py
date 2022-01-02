import numpy as np


def distance(x0, x1):
    """Compute distance between two vectors x0, x1 using the dot product.

    Args:
    x0, x1: ndarray of shape (D,) to compute distance between.

    Returns:
    the distance between the x0 and x1.
    """
    # YOUR CODE HERE
    ### Uncomment and modify the code below
    x_diff = x0 - x1
    distance = np.sqrt(x_diff.T.dot(x_diff))
    return distance


# GRADED FUNCTION: DO NOT EDIT THIS LINE

def angle(x0, x1):
    """Compute the angle between two vectors x0, x1 using the dot product.

    Args:
    x0, x1: ndarray of shape (D,) to compute the angle between.

    Returns:
    the angle between the x0 and x1.
    """
    # YOUR CODE HERE
    ### Uncomment and modify the code below
    angles_nom = x0.T.dot(x1)
    angels_den = np.sqrt(x0.T.dot(x0) * x1.T.dot(x1))
    # print(angles_nom, " # ", angels_den)
    radians = np.arccos(angles_nom / angels_den)
    angle = radians
    return angle


a = np.array([1, 0])
b = np.array([0, 1])
print(distance(a, b))
print(angle(a, b) / (np.pi * 2) * 360.)
