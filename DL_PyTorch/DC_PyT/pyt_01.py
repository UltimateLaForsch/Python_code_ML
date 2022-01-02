# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 22:40:40 2021

@author: Ultimate LaForsch
"""

import torch
import sklearn

x = torch.rand(1000, 1000)
y = torch.rand(1000, 1000)
z = torch.rand(1000, 1000)
q = torch.matmul(x, y) 
f = z * q
mean_f = torch.mean(f)
mean_f.backward() 


print(mean_f)
print("Gradients: ", x.grad, " ", y.grad, " ", z.grad)

