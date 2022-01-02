# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 17:17:19 2021

@author: Ultimate LaForsch
"""

import torch
import torch.nn.functional as F

im = torch.tensor([[[[ 8.,  1.,  2.,  5.,  3.,  1.],
          [ 6.,  0.,  0., -5.,  7.,  9.],
          [ 1.,  9., -1., -2.,  2.,  6.],
          [ 0.,  4.,  2., -3.,  4.,  3.],
          [ 2., -1.,  4., -1., -2.,  3.],
          [ 2., -4.,  5.,  9., -7.,  8.]]]])

# Build a pooling operator with size `2`.
max_pooling = torch.nn.MaxPool2d(2)

# Apply the pooling operator
output_feature = max_pooling(im)

# Use pooling operator in the image
output_feature_F = F.max_pool2d(im, 2)

# print the results of both cases
print('Max pooling:')
print(output_feature)
print(output_feature_F)

# Build a pooling operator with size `2`.
avg_pooling = torch.nn.AvgPool2d(2)

# Apply the pooling operator
output_feature = avg_pooling(im)

# Use pooling operator in the image
output_feature_F = F.avg_pool2d(im, 2)

# print the results of both cases
print('Avg pooling:')
print(output_feature)
print(output_feature_F)