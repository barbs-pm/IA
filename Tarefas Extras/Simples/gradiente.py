# -*- coding: utf-8 -*-

import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
        return sigmoid(x) * (1 - sigmoid(x))

learnrate = 0.5
x = np.array([3, 1, 2, 4])
y = np.array(0.5)
b = 0.5

# Pesos iniciais
w = np.array([0.1, -0.2, 0.3, 0.4])

h = np.dot(x, w)+b

nn_output = sigmoid(h)
print(nn_output)

# TODO: erro Calcular de rede neural 
error = y - nn_output
print(error)

# TODO: Calcule o termo de erro 
error_term = error * sigmoid_prime(h)

# TODO: Calcule a mudança nos pesos
del_w = learnrate * error_term * x

print(del_w)

w = w + del_w

print(w)

h = np.dot(x, w)+b

nn_output = sigmoid(h)
print(nn_output)

print('Neural Network output:')
print(nn_output)
print('Amount of Error:')
print(error)
print('Change in Weights:')
print(del_w)
