import numpy as np
import pandas as pd
from tqdm import tqdm

from random_decision_tree import RandomDecisionTree


class RandomForest:
    def __init__(self, train_data, test_data, attributes, labels, max_depth, num_trees, subset_size,
                 criterion='entropy'):
        self.train_data = train_data
        self.test_data = test_data
        self.attributes = attributes
        self.labels = labels
        self.max_depth = min(len(attributes), max_depth)
        self.subset_size = subset_size
        self.num_trees = num_trees
        self.criteria = criterion
        self.trees = []
        self.train_error, self.test_error = self.build_trees()

    def build_trees(self):
        train_error = []
        test_error = []
        sample_subset_size = len(self.train_data)

        for _ in tqdm(range(self.num_trees)):
            # get samples uniformly with replacement
            samples = self.train_data.sample(n=sample_subset_size, replace=True)
            # build a decision tree
            tree = RandomDecisionTree(samples, self.attributes, samples['y'], self.max_depth, self.subset_size,
                                      self.criteria)
            # add the tree to the forest
            self.trees.append(tree)
            # calculate the training error
            train_error.append(self.evaluate(self.train_data, 'y'))
            # calculate the test error
            test_error.append(self.evaluate(self.test_data, 'y'))

        return train_error, test_error

    def predictions(self, data):
        """Predict the labels of a dataset"""
        return data.apply(self.predict, axis=1)

    def predict(self, row):
        """Predict the label of a row"""
        predictions = [tree.predict(row) for tree in self.trees]
        return max(set(predictions), key=predictions.count)

    def evaluate(self, data: pd.DataFrame, label: str):
        """Evaluate the accuracy of the decision tree"""
        predictions = self.predictions(data)
        actual = data[label]
        # average prediction error
        return np.mean(predictions != actual)