import numpy as np


class Analytical_Solution:

    def __init__(self,  x_data, y_data):
        self.weight_vec = np.dot(np.linalg.inv(np.dot(x_data.T, x_data)), np.dot(x_data.T, y_data))

    def analytical_loss_func(self, x_data, y_data):
        predict_y = np.dot(x_data, self.weight_vec)
        analytical_loss = 0.5 * np.sum(np.square(y_data - predict_y))
        return analytical_loss