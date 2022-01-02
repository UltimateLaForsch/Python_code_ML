# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 15:45:30 2021

@author: Ultimate LaForsch
"""
import sys
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision 
import torch.utils.data 
import torchvision.transforms as transforms  

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
    
    
# Transform data to tensors with certain mean and stdv
transform = transforms.Compose( 
    [transforms.ToTensor(), 
     transforms.Normalize((0.1307), (0.3081))]) 

# Load data
trainset = torchvision.datasets.MNIST('mnist', train=True, 
                                        download=True, transform=transform) 

testset = torchvision.datasets.MNIST('mnist', train=False, 
                                       download=True, transform=transform)
 

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, 
                                          shuffle=True, num_workers=0) 

testloader = torch.utils.data.DataLoader(testset, batch_size=32, 
                                         shuffle=False, num_workers=0)

# Compute the shape of the training set and testing set
trainset_shape = trainloader.dataset.train_data.shape
testset_shape = testloader.dataset.test_data.shape

# Print the computed shapes
print(trainset_shape, testset_shape)

# Compute the size of the minibatch for training set and testing set
trainset_batchsize = trainloader.batch_size
testset_batchsize = testloader.batch_size

# Print sizes of the minibatch
print(trainset_batchsize, testset_batchsize)

# Instantiate the Adam optimizer and Cross-Entropy loss function
model = Net()   
optimizer = optim.Adam(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()

# Train the network
epoch = 0  
for batch_idx, data_target in enumerate(trainloader):
    print('\r' + 'Train Epoch-no. ' + str(epoch), end='')
    sys.stdout.flush()
    # print('Epoch-no. ', epoch)
    data = data_target[0]
    target = data_target[1]
    data = data.view(-1, 28 * 28)
    optimizer.zero_grad()

    # Complete a forward pass
    output = model(data)

    # Compute the loss, gradients and change the weights
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    epoch += 1

# Set the model in eval mode
model.eval()

print('')
total = 0
correct = 0
test_loop = 0
for i, data in enumerate(testloader, 0):
    print('\r' + 'Test run no. ' + str(test_loop), end='')
    sys.stdout.flush()
    # print('Test loop no. ', test_loop)
    inputs, labels = data    
    # Put each image into a vector
    inputs = inputs.view(-1, 28 * 28 * 1)    
    # Do the forward pass and get the predictions
    outputs = model(inputs)
    _, outputs = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (outputs == labels).sum().item()
    test_loop += 1
print('\nThe testing set accuracy of the network is: %d %%' % (100 * correct / total))




