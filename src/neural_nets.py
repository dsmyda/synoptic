# Author: Daniel Smyda
# Course: Stat Approaches to NLP
# Professor: Benjamin Wellner
# Date: Dec 9, 2016

from net_architecture import BasicNeuralNet
from codebook import Codebook
from random import shuffle
from hidden_layer import HiddenLayer, InputLayer, OutputLayer
import numpy as np

class MLP(BasicNeuralNet):
    """ Vanilla implementation of a Multi-layer perceptron"""

    def train(self, instances, dev_set=None, learning_rate=0.5, max_epoch=30, batch_size=30):
        """ Supervised learning on instances list. """
        self.codebook = Codebook()
        self.codebook.supervised_populate(instances)
        self.hiddenLayerSize.insert(0, self.codebook.feature_size())
        self.hiddenLayerSize.append(self.codebook.label_size())
        self.network = []

        self.network.append(InputLayer(self.hiddenLayerSize[0], self.hiddenLayerSize[1], learning_rate))
        for n in range(1, len(self.hiddenLayerSize)-1):
            self.network.append(HiddenLayer(self.hiddenLayerSize[n], self.hiddenLayerSize[n+1], learning_rate))
        self.network.append(OutputLayer(self.codebook.label_size(), learning_rate))

        self._learn(instances, dev_set, max_epoch, batch_size)

    def _partition_dataset(self, instances, size):
        """ Partition the dataset into fixed sized blocks """

        shuffle(instances)
        for i in range(0, len(instances), size):
            yield instances[i:i + size]

    def _learn(self, instances,dev_set, max_epoch, batch_size):
        """ Gradient descent algorithm, uses minibatch gradient descent if not specified. """

        for epoch in range(1, max_epoch + 1):
            for partition in self._partition_dataset(instances, batch_size):
                self._gradient_descent(partition)
            if dev_set:
                print("(Epoch, dev accuracy):", (epoch, self.accuracy(dev_set)))

    def _gradient_descent(self, instances):
        """ Backward-propagation. Compute the gradient given some portion of the training data. Accumulate the gradient,
        and return so that weights and biases can be updated."""

        feature_matrix, label_matrix = self.data2matrices(instances)
        self._forward_propagation(feature_matrix)
        self._backpropagation(label_matrix)

    def _forward_propagation(self, feature_matrix):
        """ Forward propagate the features through the neural network. Return the forward_matrix to be used in
        backward propagation."""

        self.network[0].forward_vals = feature_matrix          # Forward values of the input layer is the feature vector, will
                                                       # propagate through the network
        for layer in range(1, len(self.network)):
            self.network[layer].forward_values(self.network[layer-1])

    def _backpropagation(self, label_matrix):

        self.network[-1].backward_pass(label_matrix)
        for layer in range(len(self.network)-2, -1, -1):
            self.network[layer].backward_pass(self.network[layer+1])

    def data2matrices(self, instances):
        features = np.zeros((len(instances), self.codebook.feature_size()))
        labels = np.zeros((len(instances), self.codebook.label_size()))

        for i, instance in enumerate(instances):
            feature_vector = [self.codebook.feature_index(feature) for feature in instance.features() if feature in self.codebook._features2index]
            features[i, feature_vector] += 1

            labels[i, self.codebook.label_index(instance.label)] += 1

        return features, labels

    def accuracy(self, instances):
        # Simple accuracy test for the dev set

        current_state = [self.classify(x) == x.label for x in instances]
        return float(sum(current_state)) / len(current_state)


    def classify(self, instance):
        """Classify an instance and return the expected label."""

        feature_matrix, label_matrix = self.data2matrices([instance])
        self._forward_propagation(feature_matrix)
        return self.codebook.get_label(np.argmax(self.network[-1].forward_vals))
