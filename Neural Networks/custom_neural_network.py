from typing import List
import numpy as np
import pandas as pd
from tqdm import tqdm


class CustomNetwork:

    def __init__(self, num_features, no_of_node, weights_initialization, d,weights, initial_learning_rate):
        self.num_features = num_features
        self.no_of_nodes = no_of_node
        self.lr = initial_learning_rate
        self.d = d
        if weights_initialization == "random":
            self.weights = [np.random.randn(num_features + 1, no_of_node), np.random.randn(no_of_node + 1, no_of_node),
                            np.random.randn(no_of_node + 1, 1)]
        elif weights_initialization == "zeros":
            self.weights = [np.zeros((num_features + 1, no_of_node)), np.zeros((no_of_node + 1, no_of_node)),
                            np.zeros((no_of_node + 1, 1))]
        else:
            self.weights = weights

        self.lr_inc = self.learning_rate_increase_on_a

    def sigmoid(self, x: np.ndarray):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x: np.ndarray):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def learning_rate_increase_on_a(self, epoch):
        return self.lr / (1 + (self.lr / self.d) * epoch)

    def train(self, x: np.ndarray, y: np.ndarray, epochs: int):
        for epoch in range(epochs):
            lr_a = self.lr_inc(epoch)

            # shuffle the data
            index = np.random.permutation(len(x))
            train_x = x[index]
            train_y = y[index]

            for i in range(len(x)):
                # forward pass
                activation_outputs = [train_x[i]]
                z = []
                for j in range(len(self.weights)):
                    input_x = np.hstack((activation_outputs[j], np.ones(1)))
                    z.append(np.dot(input_x, self.weights[j]))
                    activation_outputs.append(self.sigmoid(z[j]))

                # backward pass
                delta = [activation_outputs[-1] - train_y[i]]
                for j in range(len(self.weights) - 1, 0,
                               -1):
                    delta.append(np.dot(delta[-1], self.weights[j][:-1, :].T) * self.sigmoid_derivative(activation_outputs[j + 1]))
                delta.reverse()

                for j in range(len(self.weights)):
                    input_x = np.hstack((activation_outputs[j], np.ones(1)))
                    self.weights[j] -= lr_a * np.dot(input_x[:, np.newaxis], delta[j][np.newaxis, :])
            # loss
            loss = 0
            for i in range(len(x)):
                a = [x[i]]
                for j in range(len(self.weights)):
                    input_x = np.hstack((a[j], np.ones(1)))
                    a.append(self.sigmoid(np.dot(input_x, self.weights[j])))
                loss += (a[-1] - y[i]) ** 2
            loss /= len(x)
            #print(f"Epoch: {epoch}, Loss: {loss}")

    def predict(self, x: np.ndarray):
        for i in range(len(x)):
            a = [x[i]]  # input layer
            for j in range(len(self.weights)):

                input_x = np.hstack((a[j], np.ones(1)))
                a.append(self.sigmoid(np.dot(input_x, self.weights[
                    j])))
        if a[-1] >= 0.5:
            return 1
        else:
            return 0

    def evaluate(self, x: np.ndarray, y: np.ndarray):
        return np.mean(self.predict(x) != y)


if __name__ == '__main__':

    bank_train_df = pd.read_csv('bank-note/train.csv', header=None)
    bank_test_df = pd.read_csv('bank-note/test.csv', header=None)

    bank_train_df.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'label']
    bank_test_df.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'label']

    X_train = bank_train_df.iloc[:, :-1].values
    Y_train = bank_train_df.iloc[:, -1].values


    X_test = bank_test_df.iloc[:, :-1].values
    Y_test = bank_test_df.iloc[:, -1].values

    lr = 0.001
    d = 0.01
    T = 15
    weights = []
    no_of_nodes = [5, 10, 25, 50, 100]

    print("------ Neural Network with random initialization--------")
    for no_of_node in tqdm(no_of_nodes):
        print("Number of nodes: " + str(no_of_node))
        nn_model = CustomNetwork(X_train.shape[1], no_of_node, "random", d, weights, lr)
        nn_model.train(X_train, Y_train, T)

        print('Training Error: ' + str(nn_model.evaluate(X_train, Y_train)))
        print('Testing Error: ' + str(nn_model.evaluate(X_test, Y_test)))
        print()

    # (2c)
    print("------ Neural Network with 0 initialization--------")
    for no_of_node in tqdm(no_of_nodes):
        print("Number of nodes: " + str(no_of_node))
        nn_model = CustomNetwork(X_train.shape[1], no_of_node, "zeros", d, weights, lr)
        nn_model.train(X_train, Y_train, T)

        print('Training Error: ' + str(nn_model.evaluate(X_train, Y_train)))
        print('Testing Error: ' + str(nn_model.evaluate(X_test, Y_test)))
        print()