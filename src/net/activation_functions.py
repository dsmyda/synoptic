# Author: Daniel Smyda
# Course: Stat Approaches to NLP
# Professor: Benjamin Wellner
# Date: Dec 13, 2016

import numpy as np

# x is the vector of inputs
# w is the vector of synaptic weights
# b is the vector of biases

def dot_product(x,w,b):
    return x.dot(w) + b

def tanh(val):
    return np.tanh(val)

def tanh_derivative(val):
    return 1.0 - tanh(val)**2

def efficient_tanh_derivative(val):
    return 1.0 - efficient_tanh(val)**2

def efficient_tanh(val):
    """ Try to counteract saturation around target values {-1,1}"""
    return 1.7159*tanh(val * (2.0/3.0))

def sigmoidal(val):
    return 1 / (1 + np.exp(-val))

def sigmoidal_derivative(val):
    return sigmoidal(val)*(1-sigmoidal(val))

def gaussian(x, w, b):
    #TODO implement gaussian function
    pass
