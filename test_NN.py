# Author: Daniel Smyda
# Course: Stat Approaches to NLP
# Professor: Benjamin Wellner
# Date: Dec 17, 2016

from unittest import TestCase, main
from neural_nets import MLP
from random import shuffle, seed
from competing_models import MaxEnt, NaiveBayes
from corpus import Document, ReviewCorpus, NamesCorpus, EmailCorpus

class BagOfWords(Document):

    def features(self):
        """Trivially tokenized words."""
        return self.data.split()

class Name(Document):
    def features(self):
        name = self.data
        return ['First=%s' % name[0], 'Last=%s' % name[-1]]

def accuracy(classifier, test):
    correct = [classifier.classify(x) == x.label for x in test]
    print("%.2d%% " % (100 * sum(correct) / len(correct)))
    return float(sum(correct)) / len(correct)

#class NaiveBayesTest(TestCase):

    #def split_emails_corpus(self, document_class=BagOfWords):
    #    emails = EmailCorpus(document_class=document_class)
    #    seed(hash("emails"))
    #    shuffle(emails)
    #    return (emails[:8000], emails[8000:9000], emails[9000:])

    #def test_emails(self):
    #    train, dev, test = self.split_emails_corpus()
    #    classifier = NaiveBayes()
    #    classifier.train(train)
    #    acc = accuracy(classifier, test)
    #    self.assertGreater(acc, .80)

class MaxEntTest(TestCase):
    #uTests for the MaxEnt classifier.

    #def split_names_corpus(self, document_class=Name):
    #    names = NamesCorpus(document_class=document_class)
    #    self.assertEqual(len(names), 5001 + 2943) # see names/README
    #    seed(hash("names"))
    #    shuffle(names)
    #    return (names[:5000], names[5000:6000], names[6000:])

    #def test_names_nltk(self):
    #    train, dev, test = self.split_names_corpus()
    #    classifier = MaxEnt()
    #    classifier.train(train)
    #    acc = accuracy(classifier, test)
    #    self.assertGreater(acc, 0.70)

    def split_emails_corpus(self, document_class=BagOfWords):
        emails = EmailCorpus(document_class=document_class)
        seed(hash("emails"))
        shuffle(emails)
        return (emails[:8000], emails[8000:9000], emails[9000:])

    def test_emails(self):
        train, dev, test = self.split_emails_corpus()
        max_ent = MaxEnt()
        max_ent.train(train, dev, max_epoch=25, batch_size=1, learning_rate=.5)
        acc = accuracy(max_ent, test)
        self.assertGreater(acc, .80)

    #def split_review_corpus(self, document_class):
    #    reviews = ReviewCorpus('yelp_reviews.json', document_class=document_class)
    #    seed(hash("reviews"))
    #    shuffle(reviews)
    #    return (reviews[:10000], reviews[10000:14000])

    #def test_reviews_bag(self):
    #    train, test = self.split_review_corpus(BagOfWords)
    #    classifier = MaxEnt()
    #    classifier.train(train)
    #    self.assertGreater(accuracy(classifier, test), 0.55)

class TestNN(TestCase):

    #def split_names_corpus(self, document_class=Name):
    #    """Split the names corpus into training, dev, and test sets"""
    #    names = NamesCorpus(document_class=document_class)
    #    self.assertEqual(len(names), 5001 + 2943) # see names/README
    #    seed(hash("names"))
    #    shuffle(names)
    #    return (names[:5000], names[5000:6000], names[6000:])

    #def test_names_nltk(self):
    #    """Classify names using NLTK features"""
    #    train, dev, test = self.split_names_corpus()
    #    classifier = MLP()
    #    classifier.train(train, dev)
    #    acc = accuracy(classifier, test)
    #    self.assertGreater(acc, 0.70)

    def split_emails_corpus(self, document_class=BagOfWords):
        emails = EmailCorpus(document_class=document_class)
        seed(hash("emails"))
        shuffle(emails)
        return (emails[:8000], emails[8000:9000], emails[9000:])

    def test_emails(self):
        train, dev, test = self.split_emails_corpus()
        neural_net = MLP(hiddenLayerSizes=[12,4])
        neural_net.train(train, dev, max_epoch=25, batch_size=1, learning_rate=.5)
        acc = accuracy(neural_net, test)
        self.assertGreater(acc, .80)

    #def split_review_corpus(self, document_class):
    #    """Split the yelp review corpus into training, dev, and test sets"""
    #    reviews = ReviewCorpus('yelp_reviews.json', document_class=document_class)
    #    seed(hash("reviews"))
    #    shuffle(reviews)
    #    return (reviews[:10000], reviews[10000:11000], reviews[11000:14000])

    #def test_reviews_bag(self):
    #    """Classify sentiment using bag-of-words"""
    #    train, dev, test = self.split_review_corpus(BagOfWords)
    #    classifier = MLP()
    #    classifier.train(train, dev)
    #    self.assertGreater(accuracy(classifier, test), 0.55)

if __name__ == '__main__':
    # Run all of the tests, print the results, and exit.
    main(verbosity=2)