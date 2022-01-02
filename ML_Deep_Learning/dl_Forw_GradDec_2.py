import numpy as np
import matplotlib.pyplot as plt

weights = np.array([0, 2, 1])
input_data = np.array([1, 2, 3])
target = 0
learning_rate = 0.01

def get_slope(input_data, target, weights):
    preds = weights.dot(input_data)
    error = preds - target
    slope = 2 * input_data * error
    return slope

def get_mse(input_data, target, weights):
    mse = (pow(target - (weights.dot(input_data)), 2).mean())
    return mse

n_updates = 20
mse_hist = []

# Iterate over the number of updates
for i in range(n_updates):
    # Calculate the slope: slope
    slope = get_slope(input_data, target, weights)

    # Update the weights: weights
    weights = weights - learning_rate * slope

    # Calculate mse with new weights: mse
    mse = get_mse(input_data, target, weights)

    # Append the mse to mse_hist
    mse_hist.append(mse)

# Plot the mse history
plt.plot(mse_hist)
plt.xlabel('Iterations')
plt.ylabel('Mean Squared Error')
plt.show()


