# Author: Daniel Smyda
# Course: Stat Approaches to NLP
# Professor: Benjamin Wellner
# Date: Dec 9, 2016

from net_architecture import BasicNeuralNet
from codebook import Codebook
from random import shuffle
import activation_functions
import numpy as np

class NN(BasicNeuralNet):
    """ Vanilla implementation of a Neural Net """

    def activation_function(self, w, x, b):
        """activation_functions class contains a few activation functions."""
        return activation_functions.tanh(w,x,b)

    def train(self, instances):
        """ Supervised learning on instances list. """
        self.codebook = Codebook()
        self.codebook.supervised_populate(instances)

        np.random.seed(0)

        #Input dimensions will be the number of unique words in the data
        #Output dimensions will be the number of labels we discovered while populating codebook
        self.inputWeights = np.random.rand(self.codebook.feature_size(), self.hiddenNodes)
        self.hiddenWeights = np.random.rand(self.hiddenNodes, self.hiddenLayers)
        self.outputWeights = np.random.rand(self.hiddenLayers, self.codebook.label_size())

        #Create bias matrices for the number of layers and nodes in our NN.
        #For hiddenBiases : row = nodes, col = biases for the given layer. Therefore, col vector = bias weights for entire
        #layer. Having row = nodes allows us to leverage the power of numpy operations.
        self.inputBias = np.zeros(self.hiddenNodes)
        self.hiddenBiases = np.zeros(self.hiddenNodes, self.hiddenLayers)
        self.outputBias = np.zeros(self.codebook.label_size())

        self._minibatch_gd(instances, 0.0001, 30)

    def _minibatch_gd(self, instances, learning_rate, batch_size):

        def sliced_instances():
            shuffle(instances)
            for i in range(0, len(instances), batch_size):
                yield instances[i:i + batch_size]

        for minibatch in sliced_instances():
            for instance in minibatch:
                #TODO implement backpropagation for a MLP. Backpropagate error from the output layer
                #run features forward through network
                #for each layer starting from the output layer, compute the gradient
                #update the layer weights given the gradient
                pass

    def classify(self, instance):
        """Classify an instance and return the expected label."""
        pass

class RNN(BasicNeuralNet):
    """ Recurrent Neural Net """

    def activation_function(self, w, x, b):
        pass

    def train(self, instances):
        pass

    def classify(self, instance):
        """Classify an instance and return the expected label."""
        return None