import numpy as np

def std_per_train(X,Y,epochs, lr=1):
    weights = np.zeros(X.shape[1])
    for _ in range(epochs):
        index = np.random.permutation(len(X))
        X = X[index]
        Y = Y[index]
        for i in range(len(X)):
            if Y[i] * np.dot(X[i], weights) <= 0:
                weights = weights + Y[i] * X[i] * lr
    return weights

def std_per_predict(X, weights):
    return np.sign(np.dot(X, weights))

def std_per_evaluate(X, Y, weights):
    return np.mean(std_per_predict(X, weights) != Y)
