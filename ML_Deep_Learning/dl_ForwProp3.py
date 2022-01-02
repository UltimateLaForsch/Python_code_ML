import numpy as np


def relu(activity_input):
    """Define your relu activation function here"""
    # Calculate the value for the output of the relu function: output
    output = max(0, activity_input)
    return output


def predict_with_network(input_data_row, weights):
    input_row = np.array(input_data_row)
    node_0_input = input_row.dot(weights['node_0'])
    # node_0_input = (input_data_row * weights['node_0']).sum()
    node_0_output = relu(node_0_input)
    node_1_input = input_row.dot(weights['node_1'])
    # node_1_input = (input_data_row * weights['node_1']).sum()
    node_1_output = relu(node_1_input)
    # Put node values into array: hidden_layer_outputs
    hidden_layer_outputs = np.array([node_0_output, node_1_output])
    input_to_final_layer = hidden_layer_outputs.dot(weights['output'])
    model_output = relu(input_to_final_layer)
    return model_output


input_data = [[3, 5], [1, -1], [0, 0], [8, 4]]
weights = {'node_0': np.array([2, 4]), 'node_1': np.array([4, -5]), 'output': np.array([2, 7])}
# Create empty list to store prediction results
results = []
for input_data_row in input_data:
    # Append prediction to results
    results.append(predict_with_network(input_data_row, weights))

# Print results
print(results)
