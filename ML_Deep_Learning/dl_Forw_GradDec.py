import numpy as np

weights = np.array([0, 2, 1])
input_data = np.array([1, 2, 3])
target = 0
print('Original weights: ', weights)
# Set the learning rate: learning_rate
learning_rate = 0.01
print('Learning rate: ', learning_rate)

# Calculate the predictions: preds
preds = weights.dot(input_data)

# Calculate the error: error
error = preds - target

# Calculate the slope: slope
slope = 2 * input_data * error
print('slope: ', slope)

# Update the weights: weights_updated
print('Weight update = weight - (slope * learning rate)')
weights_updated = weights - (learning_rate * slope)
print('Updated weights: ', weights_updated)

# Get updated predictions: preds_updated
preds_updated = weights_updated.dot(input_data)

# Calculate updated error: error_updated
error_updated = preds_updated - target

# Print the original error
print(error)

# Print the updated error
print(error_updated)
