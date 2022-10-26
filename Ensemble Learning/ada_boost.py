import numpy as np
import pandas as pd
from tqdm import tqdm

from Decision_Tree.ID3_algo import Decision_Tree


class Ada_Boost:
    def __init__(self, train_data, test_data, attributes, labels, max_depth, num_trees, criterion='entropy'):
        self.train_data = train_data
        self.test_data = test_data
        self.attributes = attributes
        self.labels = labels
        self.max_depth = max_depth
        self.num_trees = num_trees
        self.criteria = criterion
        self.trees = []
        self.train_error_decision_tree, self.test_error_decision_tree, \
        self.train_error, self.test_error = self.build_trees(train_data, test_data, attributes, labels, num_trees)

    def build_trees(self, train_data: pd.DataFrame, test_data: pd.DataFrame, attributes: list, labels: pd.Series,
                    num_trees: int):
        # Initialize the weights of each example to 1/m
        weights = np.ones(len(train_data)) / len(train_data)
        train_error = []
        test_error = []
        train_error_decision_tree = []
        test_error_decision_tree = []

        for _ in tqdm(range(num_trees)):
            # Create a decision tree
            # get samples uniformly with replacement
            samples = self.train_data.sample(n=self.train_data.shape[0], replace=True, weights=weights)
            # build a decision tree
            tree = Decision_Tree(samples, self.attributes, samples['y'], self.max_depth, self.criteria)
            # Calculate the predictions of the decision tree
            predictions = tree.predictions(train_data)
            # Calculate the error of the decision tree
            error = np.sum(weights[predictions != labels])
            # Calculate the weight of the decision tree
            weight = 0.5 * np.log((1 - error) / error)
            # Update the weights of the examples
            weights = weights * np.exp(-weight * labels * predictions)
            # Normalize the weights
            weights = weights / np.sum(weights)
            # Add the tree to the forest
            self.trees.append((tree, weight))
            # Calculate the training and testing error
            train_error_decision_tree.append(tree.evaluate(train_data, 'y'))
            test_error_decision_tree.append(tree.evaluate(test_data, 'y'))
            train_error.append(self.evaluate(train_data, 'y'))
            test_error.append(self.evaluate(test_data, 'y'))

        return train_error_decision_tree, test_error_decision_tree, train_error, test_error

    def predict(self, row):
        """Predict the label of a row"""
        predictions = []

        for tree, weight in self.trees:
            predictions.append(weight * tree.predict(row))

        return np.sign(sum(predictions))

    def predictions(self, data):
        """Predict the labels of a dataset"""
        return data.apply(self.predict, axis=1)

    def evaluate(self, data: pd.DataFrame, label: str):
        """Evaluate the accuracy of the decision tree"""
        predictions = self.predictions(data)
        actual = data[label]
        # average prediction error
        return np.mean(predictions != actual)