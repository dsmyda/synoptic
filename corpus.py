# Author: Daniel Smyda
# Course: Stat Approaches to NLP
# Professor: Benjamin Wellner
# Date: Dec 9, 2016

from abc import abstractmethod

class Document(object):
    """ Document class that will hold the Document text and summary. """

    def __init__(self, feature, label=None):
        self.feature = feature
        self.label = label

    def features(self):
        return self.feature

    def label(self):
        return self.label

class Corpus(object):

    def __init__(self):
        self.documents = []

    @abstractmethod
    def fetch(self, fp):
        pass

class PortugueseCorpus(Corpus):

    def fetch(self, fp):
        with open(fp, 'r') as f:
            pass