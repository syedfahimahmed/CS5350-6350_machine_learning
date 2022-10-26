import numpy as np
import pandas as pd
from tqdm import tqdm

from Decision_Tree.ID3_algo import Decision_Tree


class Bagged_Tree:
    def __init__(self, training_data, testing_data, features, labels, max_depth, num_trees, benchmark='entropy'):
        self.training_data = training_data
        self.testing_data = testing_data
        self.features = features
        self.labels = labels
        self.max_depth = max_depth
        self.num_trees = num_trees
        self.benchmark = benchmark
        self.trees = []
        self.training_error, self.testing_error = self.build_tree()

    def build_tree(self):

        training_error = []
        testing_error = []
        subset_size = len(self.training_data)

        for _ in tqdm(range(self.num_trees)):

            bootstrap = self.training_data.sample(n=subset_size, replace=True)
            tree = Decision_Tree(bootstrap, self.features, bootstrap['y'], self.max_depth, self.benchmark)
            self.trees.append(tree)

            training_error.append(self.evaluate(self.training_data, 'y'))
            testing_error.append(self.evaluate(self.testing_data, 'y'))

        return training_error, testing_error

    def predict(self, row):
        predictions = []

        for tree in self.trees:
            predictions.append(tree.predict(row))

        return max(set(predictions), key=predictions.count)

    def predictions(self, data):
        return data.apply(self.predict, axis=1)

    def evaluate(self, data: pd.DataFrame, label: str):
        predictions = self.predictions(data)
        actual = data[label]
        return np.mean(predictions != actual)