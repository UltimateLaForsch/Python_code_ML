import numpy as np

input_data = np.array([3, 5])
weights = {'node_0_0': np.array([2, 4]),
           'node_0_1': np.array([4, -5]),
           'node_1_0': np.array([-1, 2]),
           'node_1_1': np.array([1, 2]),
           'output': np.array([2, 7])}


def relu(activity_input):
    """Define your relu activation function here"""
    # Calculate the value for the output of the relu function: output
    activity_output = max(0, activity_input)
    return activity_output


def predict_with_network(input_data):
    node_0_0_input = input_data.dot(weights['node_0_0'])
    node_0_0_output = relu(node_0_0_input)
    node_0_1_input = input_data.dot(weights['node_0_1'])
    node_0_1_output = relu(node_0_1_input)
    hidden_0_outputs = np.array([node_0_0_output, node_0_1_output])
    node_1_0_input = hidden_0_outputs.dot(weights['node_1_0'])
    node_1_0_output = relu(node_1_0_input)
    node_1_1_input = hidden_0_outputs.dot(weights['node_1_1'])
    node_1_1_output = relu(node_1_1_input)
    hidden_1_outputs = np.array([node_1_0_output, node_1_1_output])
    model_output = hidden_1_outputs.dot(weights['output'])
    return model_output


output = predict_with_network(input_data)
print(output)
