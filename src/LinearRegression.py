import numpy as np


class LinearRegression:

    def __init__(self, learningRate):
        self.learningRate = learningRate
        self.weights = np.array([1, 2])
        self.bias = 2
        self.data = np.array([
            [1, 2, 7],
            [3, 4, 3],
           # [5, 6, 4],
           # [5, 3, 5]
        ])
        self.dataSize = float(len(self.data))


    def predict(self, data):
        return np.dot(self.weights, data) + self.bias

    def total_error(self):
        sumError = 0
        print(self.weights)

        for currentLine, y in zip(self.data[0:, :2],
                               self.data[:,2]):
            sumError += (self.predict(currentLine) - y)
        return sumError / self.dataSize

    def step(self):
        new_grad_weight = np.zeros(2)
        new_grad_bias = 0
        for current_data, actual in zip(self.data[0:, :2],
                                        self.data[:,2]):
            tmp = self.predict(current_data) - actual
            for i in range(len(current_data)):
                new_grad_weight[i] += 0.5 * tmp * current_data[i]
            new_grad_bias += 0.5 * tmp
        self.weights = np.subtract(self.weights,
                                   (new_grad_weight/self.dataSize) * self.learningRate)
        self.bias -= (new_grad_bias/self.dataSize) * self.learningRate

    def run(self, iterations=1):
        print(iterations)
        print(self.total_error())
        for i in range(1):
            self.step()
        print(self.total_error())
