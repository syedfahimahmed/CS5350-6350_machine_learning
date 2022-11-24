import numpy as np
from scipy.optimize import minimize, Bounds

class DualSVM:
    def __init__(self, kernel_type, gamma = 0.0, C=1.0):
        self.C = C
        self.lambdas = None
        self.w = None
        self.b = None
        self.X = None
        self.y = None
        self.gamma = gamma
        self.support_vectors = None
        self.overlapping_support_vectors = None
        if kernel_type == "linear":
            self.kernel = self.linear_kernel
        elif kernel_type == "gaussian":
            self.kernel = self.gaussian_kernel

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.X = X
        self.y = y

        self.lambdas = np.zeros(n_samples)


        def objective_function(lambdas):
            out = -np.sum(lambdas) + 0.5 * np.dot(self.lambdas, np.dot(self.lambdas.T, (self.y@self.y.T) * self.kernel(self.X, self.X)))
            return out


        constraints = ({'type': 'eq', 'fun': self.constraints})
        bounds = Bounds(0, self.C)

        initial_guess = np.zeros(n_samples)


        solution = minimize(fun=objective_function, x0=initial_guess, bounds=bounds, method='SLSQP', constraints=constraints)
        self.lambdas = solution.x
        self.support_vectors = np.where(self.lambdas > 1e-5)[0]

        self.overlapping_support_vectors = np.where((self.lambdas > 1e-5) & (self.lambdas < self.C))[0]

        self.w = np.dot(self.lambdas * self.y, self.X) # w = sum(lambdas * y * X) (dot product of lambdas, y and X)
        self.b = np.dot(self.lambdas, self.y)

    def constraints(self, lambdas):
        return np.dot(lambdas.T, self.y)

    def predict(self, X: np.ndarray):
        prediction_res = []

        for i in range(len(X)):
            prediction = np.sign(sum(self.lambdas[self.support_vectors] * self.y[self.support_vectors] * self.kernel(self.X[self.support_vectors], X[i])))
            if prediction > 0:
                prediction_res.append(1)
            else:
                prediction_res.append(-1)

        return np.array(prediction_res)

    def evaluate(self, X: np.ndarray, y: np.ndarray):
        return np.mean(self.predict(X) != y)

    def linear_kernel(self, x1, x2):
        return np.dot(x1, x2.T)

    def gaussian_kernel(self, x1: np.ndarray, x2: np.ndarray):
        return np.exp(-np.linalg.norm(x1-x2)**2 / self.gamma)