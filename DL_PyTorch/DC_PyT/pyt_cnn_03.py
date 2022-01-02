# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 18:05:03 2021

@author: Ultimate LaForsch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision 
import torch.utils.data 
import torchvision.transforms as transforms  
import sys
import pyt_cnn_03_model as model


# Transform data to tensors with certain mean and stdv
transform = transforms.Compose( 
    [transforms.ToTensor(), 
     transforms.Normalize((0.1307), (0.3081))]) 

# Load data
trainset = torchvision.datasets.MNIST('mnist', train=True, 
                                        download=True, transform=transform) 

testset = torchvision.datasets.MNIST('mnist', train=False, 
                                       download=True, transform=transform)
 

trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, 
                                          shuffle=True, num_workers=0) 

testloader = torch.utils.data.DataLoader(testset, batch_size=1, 
                                         shuffle=False, num_workers=0)

# Compute the shape of the training set and testing set
trainset_shape = trainloader.dataset.train_data.shape
testset_shape = testloader.dataset.test_data.shape

# Print the computed shapes
print('Train | Test')
print('Shape')
print(trainset_shape, testset_shape)

# Compute the size of the minibatch for training set and testing set
trainset_batchsize = trainloader.batch_size
testset_batchsize = testloader.batch_size

# Print sizes of the minibatch
print('Size of mini-batch:')
print(trainset_batchsize, testset_batchsize)

net = model.Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=net.parameters(), amsgrad=False, betas=(0.9, 0.999), eps=1e-08, \
                       lr=0.0003, weight_decay=0)

from random import uniform
EL = '\x1b[K'  # clear to end of line
CR = '\r'  # carriage return
for i, data in enumerate(trainloader, 0):
    inputs, labels = data
    optimizer.zero_grad()
    text = ('Image no. ' + str(i))
    print('\r' + text + ' is trained', end='')
    sys.stdout.flush()
    # Compute the forward pass
    outputs = net(inputs)
        
    # Compute the loss function
    loss = criterion(outputs, labels)
        
    # Compute the gradients
    loss.backward()
    
    # Update the weights
    optimizer.step()
    
    
# Iterate over the data in the test_loader
for i, data in enumerate(testloader):
  
    # Get the image and label from data
    image, label = data
    
    # Make a forward pass in the net with your image
    output = net(image)
    
    # Argmax the results of the net
    _, predicted = torch.max(output.data, 1)
    if predicted == label:
        print("Yipes, your net made the right prediction " + str(predicted))
    else:
        print("Your net prediction was " + str(predicted) + ", but the correct label is: " + str(label))    
        


