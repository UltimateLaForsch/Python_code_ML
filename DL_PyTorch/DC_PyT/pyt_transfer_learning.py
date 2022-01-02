# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 18:24:43 2021

@author: Ultimate LaForsch
"""
import torch
import torch.nn as nn
import torchvision


# 1. New model
# Create a new model
model = Net()

# Change the number of out channels
model.fc = nn.Linear(7 * 7 * 512, 26)

# Train and evaluate the model
model.train()
train_net(model, optimizer, criterion)
print("Accuracy of the net is: " + str(model.eval()))



# 2. Load parameters from old model
# Create a model using
model = Net()

# Load the parameters from the old model
model.load_state_dict(torch.load('my_net.pth'))

# Change the number of out channels
model.fc = nn.Linear(7 * 7 * 512, 26)

# Train and evaluate the model
model.train()
train_net(model, optimizer, criterion)
print("Accuracy of the net is: " + str(model.eval()))

# -> accuracy will be much higher for 2. than 1.



# 3. Download resnet

# Download resnet18
model = torchvision.models.resnet18(pretrained=True)

# Freeze all the layers bar the last one
for param in model.parameters():
    param.requires_grad = False

# Change the number of output units
model.fc = nn.Linear(512, 7)