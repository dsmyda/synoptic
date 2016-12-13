# Author: Daniel Smyda
# Course: Stat Approaches to NLP
# Professor: Benjamin Wellner
# Date: Dec 13, 2016

import numpy as np

# x is the vector of inputs
# w is the vector of synaptic weights
# b is the vector of biases

def tanh(x, w, b):
    return np.tanh(x.dot(w) + b)

def sigmoidal(x, w, b):
    return 1 / (1 + np.exp(x.dot(w) + b))

def gaussian(x, w, b):
    #TODO implement gaussian function
    pass