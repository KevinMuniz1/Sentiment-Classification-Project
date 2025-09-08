from sentiment_data import *
from utils import *
from typing import List
from collections import Counter
import math
import random

# Base Feature Extractor


class FeatureExtractor(object):
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Convert a sentence into a sparse feature vector.
        Each word → index (via indexer), then we count frequency.
        """
        feats = Counter()
        for word in sentence:
            feature = "UNI:" + word
            idx = self.indexer.index_of(feature)
            if idx == -1:  # if it has not been seen yet
                if add_to_indexer:
                    idx = self.indexer.add_and_get_index(feature)
                else:
                    continue  # dont do anything with the unseen 
            feats[idx] += 1
        return feats


# Specific Feature Extractors

class UnigramFeatureExtractor(FeatureExtractor):
    def __init__(self, indexer: Indexer):
        super().__init__(indexer)

class BigramFeatureExtractor(FeatureExtractor):
    def __init__(self, indexer: Indexer):
        super().__init__(indexer)

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        feats = Counter()
        for i in range(len(sentence) - 1):
            feature = "BI:" + sentence[i] + "_" + sentence[i+1]
            idx = self.indexer.index_of(feature)
            if idx == -1:
                if add_to_indexer:
                    idx = self.indexer.add_and_get_index(feature)
                else:
                    continue
            feats[idx] += 1
        return feats

class BetterFeatureExtractor(FeatureExtractor):
    def __init__(self, indexer: Indexer):
        super().__init__(indexer)

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        feats = Counter()
        # this add together the lowercase letters + unigrams + bigrams
        sentence = [w.lower() for w in sentence]
        # unigrams
        for word in sentence:
            feature = "UNI:" + word
            idx = self.indexer.index_of(feature)
            if idx == -1:
                if add_to_indexer:
                    idx = self.indexer.add_and_get_index(feature)
                else:
                    continue
            feats[idx] += 1
        # bigrams
        for i in range(len(sentence) - 1):
            feature = "BI:" + sentence[i] + "_" + sentence[i+1]
            idx = self.indexer.index_of(feature)
            if idx == -1:
                if add_to_indexer:
                    idx = self.indexer.add_and_get_index(feature)
                else:
                    continue
            feats[idx] += 1
        return feats



# Base Classifier

class SentimentClassifier(object):
    def predict(self, sentence: List[str]) -> int:
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, sentence: List[str]) -> int:
        return 1



# Perceptron Classifier

class PerceptronClassifier(SentimentClassifier):
    def __init__(self, weights, feat_extractor):
        self.weights = weights
        self.feat_extractor = feat_extractor

    def predict(self, sentence: List[str]) -> int:
        feats = self.feat_extractor.extract_features(sentence, add_to_indexer=False)
        score = sum(self.weights[idx] * val for idx, val in feats.items())
        return 1 if score >= 0 else 0


def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor, epochs=5) -> PerceptronClassifier:
    weights = Counter()
    for epoch in range(epochs):
        random.shuffle(train_exs)
        for ex in train_exs:
            feats = feat_extractor.extract_features(ex.words, add_to_indexer=True)
            score = sum(weights[idx] * val for idx, val in feats.items())
            pred = 1 if score >= 0 else 0
            if pred != ex.label:
                for idx, val in feats.items():
                    weights[idx] += (1 if ex.label == 1 else -1) * val
    return PerceptronClassifier(weights, feat_extractor)


# Logistic Regression Classifier

class LogisticRegressionClassifier(SentimentClassifier):
    def __init__(self, weights, feat_extractor):
        self.weights = weights
        self.feat_extractor = feat_extractor

    def predict(self, sentence: List[str]) -> int:
        feats = self.feat_extractor.extract_features(sentence, add_to_indexer=False)
        z = sum(self.weights[idx] * val for idx, val in feats.items())
        prob = 1 / (1 + math.exp(-z))
        return 1 if prob >= 0.5 else 0


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor, epochs=5, lr=0.1) -> LogisticRegressionClassifier:
    weights = Counter()
    for epoch in range(epochs):
        random.shuffle(train_exs)
        for ex in train_exs:
            feats = feat_extractor.extract_features(ex.words, add_to_indexer=True)
            z = sum(weights[idx] * val for idx, val in feats.items())
            prob = 1 / (1 + math.exp(-z))
            error = ex.label - prob
            # gradient update
            for idx, val in feats.items():
                weights[idx] += lr * error * val
    return LogisticRegressionClassifier(weights, feat_extractor)



# Training Dispatcher

def train_model(args, train_exs, dev_exs):
    if args.model == "TRIVIAL":
        return TrivialSentimentClassifier()

    if args.feats == "UNIGRAM":
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER")

    # Trains the model
    if args.model == "PERCEPTRON":
        return train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        return train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR")

