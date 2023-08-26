import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class LogisticRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0

        for it in range(self.n_iters):
            # approximate y with linear combination of weights and x, plus bias
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            # calculate cost-> Binary cross entropy loss
            cost = (
                -1
                / m
                * np.sum(y * np.log(y_predicted) + (1 - y) * np.log(1 - y_predicted))
            )
            # calculate gradient/ back propagation
            dw = (
                1 / m * np.dot(X.T, (y_predicted - y))
            )  # remove the 2 its just a scling factor
            db = 1 / m * np.sum(y_predicted - y)
            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if it % 999 == 0:
                print(f"Cost after iteration {it}: {cost}")

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)
    
    def accuracy(self, y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy
    
    def dataloader(self):
        bc = datasets.load_breast_cancer()
        X, y = bc.data, bc.target

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=1234
        )
        return X_train, X_test, y_train, y_test

