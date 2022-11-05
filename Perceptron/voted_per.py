import numpy as np

def voted_per_train(X,Y,epochs,lr=1):

    weights = np.zeros(X.shape[1], dtype=float)
    weights_list = [weights]
    count=1
    count_list = [count]
    for _ in range(epochs):
        index = np.random.permutation(len(X))
        X = X[index]
        Y = Y[index]
        for i in range(len(X)):
            if Y[i] * np.dot(X[i], weights) <= 0:
                weights = weights + Y[i] * X[i] * lr
                weights_list.append(weights)
                count_list.append(count)
                count = 1
            else:
                count+=1

    return weights_list, count_list

def voted_per_predict(X, weights, counts):
    return np.sign(np.dot(counts, np.sign(np.dot(X, weights.T)).T))

def voted_per_evaluate(X, Y, weights, counts):
    return np.mean(voted_per_predict(X, weights, counts) != Y)
