# Author: Daniel Smyda
# Course: Stat Approaches to NLP
# Professor: Benjamin Wellner
# Date: Dec 13, 2016

class Codebook:

    def __init__(self):
        self._features2index = {}
        self._index2features = {}
        self._labels2index = {}
        self._index2labels = {}

    def add_feature(self, feature):
        if feature not in self._features2index:
            self._features2index[feature] = len(self._features2index)
            self._index2features[len(self._features2index) - 1] = feature

    def add_label(self, label):
        if label not in self._labels2index:
            self._labels2index[label] = len(self._labels2index)
            self._index2labels[len(self._labels2index) - 1] = label

    def get_feature(self, i):
        return self._index2features[i]

    def get_label(self, i):
        return self._index2labels[i]

    def dimension(self):
        return self.feature_size(), self.label_size()

    def coordinates(self, feature, label):
        return self.feature_index(feature), self.label_index(label)

    def feature_index(self, feature):
        if feature in self._features2index:
            return self._features2index[feature]
        return -1

    def label_index(self, label):
        if label in self._labels2index:
            return self._labels2index[label]
        return -1

    def supervised_populate(self, instances):
        for instance in instances:
            self.add_label(instance.label)
            for feature in instance.features():
                self.add_feature(feature)

    def feature_size(self):
        return len(self._features2index)

    def label_size(self):
        return len(self._labels2index)

    def get_label_codebooks(self):
        return self._labels2index, self._index2labels

    def get_feature_codebooks(self):
        return self._features2index, self._index2features