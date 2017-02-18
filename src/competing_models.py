# Author: Daniel Smyda
# Course: Stat Approaches to NLP
# Professor: Benjamin Wellner
# Date: Dec 9, 2016

import numpy as np
from scipy.misc import logsumexp
from random import shuffle
from codebook import Codebook
import math

class NaiveBayes:
    def __init__(self, alpha=1.0):
        self.vocab_lookup = {}
        self.labels_lookup = {}
        self.inverse_labels_lookup = {}
        self.prior_table = None
        self.feature_matrix = []

        # Smoothing parameters
        self.alpha = alpha
        self.lambd = 0.5

        self.label_index = 0
        self.feature_index = 0

        self.likelihood = 0.0
        self.prior = 0.0

    def discoverLabelSpace(self, instances):
        u""" I must know label space size ahead of time to populate the n-d matrix of counts, otherwise trying to
            do this dynamically with one pass over the data would be messy. Therefore, I traverse the label space
            before proceeding, which introduces little slowdown. """

        for instance in instances:
            if instance.label not in self.labels_lookup and instance.label != "":  # Ignores blank entry in blogs corpus
                self.labels_lookup[instance.label] = self.label_index
                self.inverse_labels_lookup[self.label_index] = instance.label
                self.label_index += 1


    def populate_feature_matrix(self, label, features):
        u""" Begin dynamically adding features counts and populating the feature matrix as the data is seen. At the
            same time, begin caching feature name indices in a lookup table and construct the priors count table."""

        for feature in set(features):
            if feature not in self.vocab_lookup:
                self.vocab_lookup[feature] = self.feature_index
                self.feature_matrix.append([0.0] * len(self.labels_lookup))
                self.feature_index += 1

            self.feature_matrix[self.vocab_lookup[feature]][self.labels_lookup[label]] += 1

        self.prior_table[self.labels_lookup[label]] += 1

    def train(self, instances):
        u""" Discovers label space and then populates the feature matrix and prior matrix with counts. To make
            computation easier, feature matrix is passed into numpy and all counts are converted to probability
            estimates that will be used in label prediction. Populating and adding in a list of lists is much faster to
            do than in a numpy array. """

        self.discoverLabelSpace(instances)
        self.prior_table = np.zeros(len(self.labels_lookup))

        for instance in instances:
            if instance.label != '':  # Ignores blank entry in blogs corpus
                self.populate_feature_matrix(instance.label, instance.features())

        self.feature_matrix = np.array(self.feature_matrix)
        self.feature_matrix = (self.feature_matrix + self.alpha) / (
        self.prior_table + self.alpha * self.feature_matrix.shape[0])  # Laplace Smoothing
        # self.JM_smooth_entries(len(instances))                                                                                                #Jelinek-Mercer (JM) Smoothing
        self.prior_table = self.prior_table / len(instances)


    def update_likelihood(self, feature, y):
        u""" Updates the likelihood by adding the log of the probability in the feature matrix. If the feature is not in
            the trained vocabulary, then just add the Laplacian."""

        if feature in self.vocab_lookup:
            self.likelihood += math.log(self.feature_matrix[self.vocab_lookup[feature], y])
        else:
            self.likelihood += math.log(self.alpha / self.alpha * self.feature_matrix.shape[0])  # Laplacian


    def classify(self, instance):
        u""" Maximizes the log-likehood across all the labels to output its best prediction. """

        args = []

        for y in range(self.prior_table.shape[0]):
            self.likelihood = self.prior_table[y]
            for feature in instance.features():
                self.update_likelihood(feature, y)

            args.append((self.likelihood, y))

        return self.inverse_labels_lookup[max(args)[1]]

class MaxEnt:
    # -*- mode: Python; coding: utf-8 -*-

    def __init__(self):
        self.parameters = np.zeros((0,0))

    def train(self, instances, dev_set=None, max_epoch=30, learning_rate=.5, batch_size=30):
        # Construct a statistical model from labeled instances.

        self.codebook = Codebook()
        self.codebook.supervised_populate(instances)

        self.parameters = np.zeros((self.codebook.dimension()))
        self._train_sgd(instances, dev_set, max_epoch, learning_rate, batch_size)

    def _mini_batch(self, instances, batch_size):
        # Yield mini-batches from the original data

        shuffle(instances)
        for i in range(0, len(instances), batch_size):
            yield instances[i:i + batch_size]

    def _compute_gradient(self, batch):
        # Compute the gradient given the current batch of data

        log_likelihood = 0
        observed_count = np.zeros(self.codebook.dimension())
        expected_count = np.zeros(self.codebook.dimension())

        for datapoint in batch:
            feature_map = [self.codebook.feature_index(feature) for feature in datapoint.features()]

            observed_count[feature_map, self.codebook.label_index(datapoint.label)] += 1
            lambda_vector = self.parameters[feature_map, :].sum(0)
            log_likelihood -= sum(lambda_vector) - logsumexp(lambda_vector)
            posterior = np.exp(lambda_vector[self.codebook.label_index(datapoint.label)] - logsumexp(lambda_vector))
            expected_count[feature_map, self.codebook.label_index(datapoint.label)] += posterior

        return observed_count - expected_count, log_likelihood

    def _train_sgd(self, train_instances, dev_set, max_epoch, learning_rate, batch_size):
        # Train MaxEnt model with Mini-batch Gradient Descent

        for epoch in range(1, max_epoch+1):
            for batch in self._mini_batch(train_instances, batch_size):
                gradient, log_likelihood = self._compute_gradient(batch)
                self.parameters += gradient * learning_rate
            if dev_set:
                print("(Epoch, accuracy):", (epoch, self.accuracy(dev_set)))

    def accuracy(self, instances):
        # Simple accuracy test for the dev set

        current_state = [self.classify(x) == x.label for x in instances]
        return float(sum(current_state)) / len(current_state)

    def classify(self, instance):
        feature_map = [self.codebook.feature_index(feature) for feature in instance.features() if feature in self.codebook._features2index]

        lambda_vector = self.parameters[feature_map, :].sum(0)
        posteriors = np.exp(lambda_vector - logsumexp(lambda_vector))
        return self.codebook.get_label(np.argmax(posteriors))
