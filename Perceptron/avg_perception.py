import numpy as np

def avg_per_train(X,Y,epochs, lr=1):
    weights = np.zeros(X.shape[1])
    a = np.zeros(X.shape[1])
    for _ in range(epochs):
        index = np.random.permutation(len(X))
        X = X[index]
        Y = Y[index]
        for i in range(len(X)):
            if Y[i] * np.dot(X[i], weights) <= 0:
                weights = weights + Y[i] * X[i] * lr
            a = a + weights
    return a, weights

def avg_per_predict(X, a):
    return np.sign(np.dot(X, a))

def avg_per_evaluate(X, Y, a):
    return np.mean(avg_per_predict(X, a) != Y)