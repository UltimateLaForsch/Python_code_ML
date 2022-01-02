import numpy as np


def relu(node_input):
    """Define your relu activation function here"""
    # Calculate the value for the output of the relu function: output
    output = max(0, node_input)
    return output


input_data = np.array([3, 5])
weights = {'node_0': np.array([2, 4]), 'node_1': np.array([4, -5]), 'output': np.array([2, 7])}

# Calculate node 0 value: node_0_output
# node0_input = (input_data * weights['node_0']).sum()
node0_input = input_data.dot(weights['node_0'])
node0_output = relu(node0_input)

# Calculate node 1 value: node_1_output
# node1_input = (input_data * weights['node_1']).sum()
node1_input = input_data.dot(weights['node_1'])
node1_output = relu(node1_input)

# Put node values into array: hidden_layer_outputs
hidden_layer_outputs = np.array([node0_output, node1_output])

# Calculate model output (do not apply relu)
# model_output = (hidden_layer_outputs * weights['output']).sum()
model_output = hidden_layer_outputs.dot(weights['output'])

# Print model output
print(model_output)