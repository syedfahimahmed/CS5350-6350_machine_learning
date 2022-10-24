import numpy as np
import pandas as pd


class Batch_Gradient_Descent:

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.weight_vec = None

    def optimize(self, x_data, y_data):

        num_data, num_features = x_data.shape
        self.weight_vec = np.zeros(num_features)

        bgd_cost = []
        norm_weight_difference = 1

        while norm_weight_difference > 0.000001:
            predict_y = np.dot(x_data, self.weight_vec)
            bgd_loss = 0.5 * np.sum(np.square(y_data - predict_y))
            delta = np.dot(x_data.T, (predict_y - y_data))

            norm_weight_difference = np.linalg.norm(self.weight_vec - (self.weight_vec - self.learning_rate * delta))
            self.weight_vec -= self.learning_rate * delta
            bgd_cost.append(bgd_loss)

        return bgd_cost

    def bgd_loss_func(self, x_data, y_data):
        predict_y = np.dot(x_data, self.weight_vec)
        bgd_loss = 0.5 * np.sum(np.square(y_data - predict_y))
        return bgd_loss