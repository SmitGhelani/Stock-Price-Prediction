import numpy as np


class LinearRegression:

    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.slope = None
        self.c = None

    def train(self, x, y):
        n_samples, n_features = x.shape
        self.slope = np.zeros(n_features)
        self.c = 0

        for _ in range(self.n_iters):
            y_predicted = np.dot(x, self.slope) + self.c
            dw = (1 / n_samples) * np.dot(x.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            self.slope -= self.lr * dw
            self.c -= self.lr * db

    def predict(self, x):
        y_approx = np.dot(x, self.slope) + self.c
        return y_approx
