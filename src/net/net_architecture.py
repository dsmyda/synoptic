# Author: Daniel Smyda
# Course: Stat Approaches to NLP
# Professor: Benjamin Wellner
# Date: Dec 9, 2016

from abc import abstractmethod
from pickle import dump, load, HIGHEST_PROTOCOL as HIGHEST_PICKLE_PROTOCOL

class BasicNeuralNet(object):
    """ Basic framework for the different neural networks classes"""

    def __init__(self, hiddenLayerSizes = []):
        """ List of hiddenLayer sizes in sequential order. """
        self.hiddenLayerSize = hiddenLayerSizes

    def get_model(self): return None
    def set_model(self, model): pass

    @abstractmethod
    def train(self, instances):
        pass

    @abstractmethod
    def classify(self, instance):
        """Classify an instance and return the expected label."""
        return None

    def save(self, file):
        """Save the current model to the given file."""
        if isinstance(file, str):
            with open(file, "wb") as file:
                self.save(file)
        else:
            dump(self.model, file, HIGHEST_PICKLE_PROTOCOL)

    def load(self, file):
        """Load a saved model from the given file."""
        if isinstance(file, str):
            with open(file, "rb") as file:
                self.load(file)
        else:
            self.model = load(file)
