# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 17:18:43 2021

@author: Ultimate LaForsch
"""

import torch
import torch.nn as nn

# Convolution operator - OOP way
images = torch.rand(10, 1, 28, 28)
print(images)
print('\nImages shape ', images.shape, '\n')

conv_filters = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, \
                          stride=1, padding=1)
output_feature = conv_filters(images)
print('Output shape ', output_feature.shape)  