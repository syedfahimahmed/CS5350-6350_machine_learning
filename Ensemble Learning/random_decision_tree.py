import numpy as np
import pandas as pd


class RandomDecisionTree:
    def __init__(self, data, attributes, labels, max_depth, subset_size, criterion='entropy'):
        self.data = data
        self.attributes = attributes
        self.labels = labels
        self.max_depth = max_depth
        self.criteria = criterion
        self.subset_size = subset_size
        self.tree = self.build_tree(data, attributes, labels)

    def build_tree(self, data: pd.DataFrame, attributes: list, labels: pd.Series, depth=0):
        # If all examples have same label, return the leaf node with the label
        if len(np.unique(labels)) == 1:
            return labels.iloc[0]

        # If there are no more attributes to split on, return the most common label(target) value
        if len(attributes) == 0:
            return np.unique(labels).tolist()[0]

        # If the depth of the tree is equal to the maximum depth, return the most common label(target) value
        if depth == self.max_depth:
            return np.unique(labels).tolist()[0]

        # get samples uniformly with replacement
        samples = data.sample(n=self.subset_size, replace=True)

        best_attribute = self.choose_attribute(samples, samples['y'], attributes)
        tree = {best_attribute: {}}  # Create a root node

        for value in set(data[best_attribute]):
            # Split the data and target based on the value of the best attribute
            new_data = data[data[best_attribute] == value]
            new_label = labels[data[best_attribute] == value]
            # Remove the best attribute from the list of attributes
            new_attributes = list(attributes[:])
            new_attributes.remove(best_attribute)
            # Create a subtree for the current value of the attribute
            subtree = self.build_tree(new_data, new_attributes, new_label, depth + 1)
            # Add the subtree to the tree under the root node
            tree[best_attribute][value] = subtree

        return tree

    def choose_attribute(self, data: pd.DataFrame, labels: pd.Series, attributes: list):
        # Calculate the information gain for each attribute
        gains = []

        for attribute in attributes:
            gains.append(self.information_gain(data, labels, attribute))

        # Return the attribute with the maximum information gain
        return attributes[gains.index(max(gains))]

    def information_gain(self, data: pd.DataFrame, labels: pd.Series, attribute: str):
        # Calculate the first term of Gain of the target values
        if self.criteria == 'entropy':
            first_term = self.entropy(labels)
        elif self.criteria == 'gini':
            first_term = self.gini_index(labels)
        else:
            first_term = self.majority_error(labels)

        # Calculate the values and the corresponding counts for the selected attribute
        values, counts = np.unique(data[attribute], return_counts=True)
        # Calculate the weighted entropy for the attribute
        second_term = 0

        for value, count_val in zip(values, counts):
            if self.criteria == 'entropy':
                second_term += (count_val / len(data)) * self.entropy(labels[data[attribute] == value])
            elif self.criteria == 'gini':
                second_term += (count_val / len(data)) * self.gini_index(labels[data[attribute] == value])
            else:
                second_term += (count_val / len(data)) * self.majority_error(labels[data[attribute] == value])

        # Calculate the information gain
        return first_term - second_term

    def entropy(self, label: pd.Series):
        # Calculate the frequency of each unique value in the target
        _, counts = np.unique(label, return_counts=True)
        # Calculate the entropy
        entropy = 0

        for count in counts:
            entropy += (-count / len(label)) * np.log2(count / len(label))

        return entropy

    def gini_index(self, label):
        # Calculate the frequency of each unique value in the target
        _, counts = np.unique(label, return_counts=True)
        # Calculate the gini index
        gini = 1

        for count in counts:
            gini -= (count / len(label)) ** 2

        return gini

    def majority_error(self, label):
        # Calculate the frequency of each unique value in the target
        _, counts = np.unique(label, return_counts=True)
        # Calculate the majority error
        majority_error = 1 - max(counts) / len(label)

        return majority_error

    def predict(self, row):
        """Predict the label of a row"""
        node = self.tree  # Start at the root node

        while isinstance(node, dict):  # While the node is not a leaf
            attribute = list(node.keys())[0]  # Get the attribute
            attribute_value = row[attribute]  # Get the value of the attribute

            if attribute_value not in node[attribute].keys():
                return None

            node = node[attribute][attribute_value]  # Move to the next node

        return node  # Return the leaf node

    def predictions(self, data):
        """Predict the labels of a dataset"""
        return data.apply(self.predict, axis=1)

    def evaluate(self, data: pd.DataFrame, label: str):
        """Evaluate the accuracy of the decision tree"""
        predictions = self.predictions(data)
        actual = data[label]
        # average prediction error
        return np.mean(predictions != actual)

    def training_error(self, label: str):
        """Calculate the training error of the decision tree"""
        return self.evaluate(self.data, label)