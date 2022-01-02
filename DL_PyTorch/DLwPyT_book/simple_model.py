# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 09:15:02 2021

@author: Ultimate LaForsch
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

def model(t_u, w, b):
    t_pred = t_u * w + b
    return t_pred

def loss_fn(t_pred, t_c):
    loss_vector = ((t_pred - t_c) ** 2)
    loss = loss_vector.mean()
    return loss

def d_loss_fn(t_p, t_c):
    d_loss = 2 * (t_p - t_c) / t_p.size(0)
    return d_loss

def d_model_dw(t_u, w, b):
    return t_u
 
def d_model_db(t_u, w, b):
    return 1.0    

def grad_fn(t_u, t_c, t_p, w, b):
    d_loss_dtp = d_loss_fn(t_p, t_c)
    d_loss_dw = d_loss_dtp * d_model_dw(t_u, w, b)
    d_loss_db = d_loss_dtp * d_model_db(t_u, w, b)
    return torch.stack([d_loss_dw.sum(), d_loss_db.sum()]) 

def training_loop(n_epochs, learning_rate, t_u, t_c, params):
    for epoch in range(1, n_epochs):        
        w, b = params
        t_pred = model(t_u, w, b)
        loss = loss_fn(t_pred, t_c)
        gradient = grad_fn(t_u, t_c, t_pred, w, b)
        print('Epoch: ', epoch, 'Params: ', params, ' | Gradient: ', gradient,
              ' | Loss: ', round(float(loss), 2))
        params = params - learning_rate * gradient       
    return params
        
def training_loop_2(n_epochs, learning_rate, t_u, t_c, params):
    for epoch in range(1, n_epochs + 1):
        if params.grad is not None:
            params.grad.zero_() 
        t_p = model(t_u, *params)
        loss = loss_fn(t_p, t_c)
        loss.backward()
 
        with torch.no_grad():      
            params -= learning_rate * params.grad
 
        if epoch % 500 == 0:
            print('Epoch %d, Loss %f' % (epoch, float(loss))) 
    return params

t_c = [0.5,  14.0, 15.0, 28.0, 11.0,  8.0,  3.0, -4.0,  6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)
simple_scaler = 0.1
t_u = t_u * simple_scaler
w = torch.ones(())
b = torch.zeros(())
t_pred = model(t_u, w, b)
print(t_pred)
loss = loss_fn(t_pred, t_c)
print(loss)

delta = 0.1877
 
loss_delta_up = loss_fn(model(t_u, w + delta, b), t_c)
loss_delta_down = loss_fn(model(t_u, w - delta, b), t_c)
print('delta up: ', loss_delta_up)
print('delta down: ', loss_delta_down)
print('delta used: ', delta)
loss_rate_of_change_w = loss_delta_up - loss_delta_down \
    / (2.0 * delta)
print('loss rate of change: ', loss_rate_of_change_w)
if loss_rate_of_change_w > 0:
    print('decrease delta!')
else:
    print('increase delta!')

check = d_loss_fn(t_pred, t_c)
print('d Loss / d t_pred: ')
print(check)
gradient = grad_fn(t_u, t_c, t_pred, w, b)    
print('Gradient: dL/dw    |  dL/db')
print(gradient)
no_epochs = 5000
learning_rate = 1e-2
params = torch.tensor([1.0, 0.0])
solution = training_loop(no_epochs, learning_rate, t_u, t_c, params)
print('Solution: ', solution, ' after ' , no_epochs, ' epochs')

t_p = model(t_u, *solution)
fig = plt.figure(dpi=600)
plt.xlabel("Temperature (°Fahrenheit)")
plt.ylabel("Temperature (°Celsius)")
plt.plot(t_u.numpy() / simple_scaler , t_p.detach().numpy())
plt.plot(t_u.numpy() / simple_scaler , t_c.numpy(), 'o')

# Let gradients compute from PyTorch
params_2 = torch.tensor([1.0, 0.0], requires_grad=True)
solution_2 = training_loop_2(no_epochs, learning_rate, t_u, t_c, params_2)
print('Solution 2: ', solution_2, ' after ' , no_epochs, ' epochs')
