# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 16:55:55 2021

@author: Ultimate LaForsch
"""

import torch
import torch.nn.functional as F

# Convolution operator - Functional way
images = torch.rand(10, 1, 28, 28)

conv_filter = torch.rand(6, 1, 3, 3)

output_feature = F.conv2d(images, conv_filter, stride=1, padding=0)
print('Output shape ', output_feature.shape)    
