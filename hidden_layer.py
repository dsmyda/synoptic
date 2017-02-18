# Author: Daniel Smyda
# Course: Stat Approaches to NLP
# Professor: Benjamin Wellner
# Date: Dec 17, 2016

import activation_functions as af
import numpy as np


class _Layer:

    def __init__(self, nodes, learning_rate, activation_function, activation_derivative):
        self.nodes = nodes
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative
        self.bias = 0
        self.learning_rate = learning_rate
        self.forward_vals = np.zeros((0,0))

    def forward_values(self, prev_layer):
        self.sums = af.dot_product(
            prev_layer.forward_vals,
            prev_layer.weights,
            self.bias
        )

        self.forward_vals = self.activation_function(self.sums)


class HiddenLayer(_Layer):
    """ Adds more flexibility to the model, can allow for layers with varying amounts of hidden nodes. """

    def __init__(self, hiddenNodes, nextLayerNodes, learning_rate, activation_function=af.sigmoidal, activation_derivative=af.sigmoidal_derivative):
        super().__init__(hiddenNodes, learning_rate, activation_function, activation_derivative)
        np.random.seed(0)

        self.nextLayerNodes = nextLayerNodes
        self.weights = np.random.uniform(low=0, high=1, size=(hiddenNodes, nextLayerNodes))

    def backward_pass(self, next_layer):
        #propagate the error delta backwards through the network
        self.delta = np.multiply(np.dot(next_layer.delta, self.weights.T), self.activation_derivative(self.sums))
        self.gradient = np.dot(self.forward_vals.T, next_layer.delta)
        self.weights += np.multiply(self.gradient, self.learning_rate)


class InputLayer(HiddenLayer):

    def __init__(self, nodes, nextLayerNodes, learning_rate, activation_function=af.sigmoidal, activation_derivative=af.sigmoidal_derivative):
        super().__init__(nodes, nextLayerNodes, learning_rate, activation_function,activation_derivative)

    def backward_pass(self, next_layer):
        self.delta = np.multiply(np.dot(next_layer.delta, self.weights.T), self.activation_derivative(self.forward_vals))
        self.gradient = np.dot(self.forward_vals.T, next_layer.delta)
        self.weights += np.multiply(self.gradient, self.learning_rate)

class OutputLayer(_Layer):

    def __init__(self, nodes, learning_rate, activation_function=af.sigmoidal, activation_derivative=af.sigmoidal_derivative):
        super().__init__(nodes, learning_rate, activation_function, activation_derivative)

    def backward_pass(self, observed_labels):
        self.delta = np.multiply((observed_labels - self.forward_vals), self.activation_derivative(self.sums))