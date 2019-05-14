import numpy as np


class LinearRegression:

    def __init__(self, data, valueToPredict, learningRate):
        self.dataSize = float(len(data))

        self.learningRate = learningRate
        self.valueToPredict = valueToPredict
        self.bias = 0

        self.Y = data[valueToPredict]
        self.X = data.drop(valueToPredict, 1).values

        self.weights = np.zeros(self.X.shape[1])


    def predict(self, data):
        return np.dot(self.weights, data) + self.bias

    def total_error(self):
        sumError = 0
        for currentLine, y in zip(self.X, self.Y):
            sumError += 0.5 * (self.predict(currentLine) - y)
        return sumError / self.dataSize

    def step(self):
        new_grad_weight = np.zeros(len(self.weights))
        new_grad_bias = 0
        for current_data, actual in zip(self.X, self.Y):
            tmp = self.predict(current_data) - actual
            for i in range(len(current_data)):
                new_grad_weight[i] += tmp * current_data[i]
            new_grad_bias += tmp
        self.weights = np.subtract(self.weights,
                                   (new_grad_weight/self.dataSize) * self.learningRate)
        self.bias -= (new_grad_bias/self.dataSize) * self.learningRate

    def run(self, iterations=10):
        for i in range(iterations):
            self.step()
