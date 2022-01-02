# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 17:40:19 2021

@author: Ultimate LaForsch
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision 
import torch.utils.data 
import torchvision.transforms as transforms  

# Example without data!

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()       
        # Instantiate all 2 linear layers  
        self.fc1 = nn.Linear(28 * 28 * 1, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):      
        # Do the forward pass
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Net2(nn.Module):
    def __init__(self):
        
        # Define all the parameters of the net
        self.classifier = nn.Sequential(
            nn.Linear(28*28, 200),
            nn.ReLU(inplace=True),
            # applying Dropout
            nn.Dropout(strength=0.5),
            nn.Linear(200, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 10)
            )
        
        def forward(self, x):
    
            # Do the forward pass
            return self.classifier(x)
        
class Net3(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Implement the sequential module for feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(10),
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(20)
            )
        
        # Implement the fully connected layer for classification
        self.fc = nn.Linear(in_features=7 * 7 * 20, out_features=10)
        
        
        
    
# Example without data!
    
model = Net()   
criterion = nn.CrossEntropyLoss()
# weight_decay -> L2-regularization
optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.001)
