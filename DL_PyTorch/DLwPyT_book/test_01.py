# -*- coding: utf-8 -*-
import torch
import numpy as np

print("PyTorch test: Generate random 5x3 tensor: \n")
x = torch.rand(5, 3)
print(x, "\n")
print("CUDA available for PyTorch: ", torch.cuda.is_available())

your_first_tensor = torch.rand(3, 3)
tensor_size = your_first_tensor.shape
print(your_first_tensor)
print(tensor_size)

# Create a matrix of ones with shape 3 by 3
tensor_of_ones = torch.ones(3, 3)

# Create an identity matrix with shape 3 by 3
identity_tensor = torch.eye(3)

# Do a matrix multiplication of tensor_of_ones with identity_tensor
matrices_multiplied = torch.matmul(tensor_of_ones, identity_tensor)
print(matrices_multiplied)

# Do an element-wise multiplication of tensor_of_ones with identity_tensor
element_multiplication = tensor_of_ones * identity_tensor
print(element_multiplication)

