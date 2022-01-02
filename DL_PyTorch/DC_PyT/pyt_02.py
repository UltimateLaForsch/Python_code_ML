# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 17:50:55 2021

@author: Ultimate LaForsch
"""

import torch 
 
x = torch.tensor(4., requires_grad=True) 
y = torch.tensor(-3., requires_grad=True) 
z = torch.tensor(5., requires_grad=True) 
 
q = x + y 
f = q * z 
 
f.backward() 
  
print("Gradient of x is: " + str(x.grad)) 
print("Gradient of y is: " + str(y.grad)) 
print("Gradient of z is: " + str(z.grad)) 
