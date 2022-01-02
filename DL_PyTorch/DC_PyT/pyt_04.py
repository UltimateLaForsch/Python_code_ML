# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 15:20:01 2021

@author: Ultimate LaForsch
"""

import torch
import torch.nn as nn

# Initialize the scores and ground truth
logits = torch.tensor([[-1.2, 0.12, 4.8]])
ground_truth = torch.tensor([2])

# Instantiate cross entropy loss
criterion = nn.CrossEntropyLoss()

# Compute and print the loss
loss = criterion(logits, ground_truth)
print(loss)

# Initialize logits and ground truth
logits = torch.rand(1, 1000)
ground_truth = torch.tensor([111])

# Instantiate cross-entropy loss
criterion = nn.CrossEntropyLoss()

# Calculate and print the loss
loss = criterion(logits, ground_truth)
print(loss)
