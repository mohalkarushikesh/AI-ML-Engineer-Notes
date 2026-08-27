import numpy as np

class Perceptron:
    def __init__(self, lr=0.01, n_iters=1000):   # __init__, not __self__
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def step_funct(self, x):
        return np.where(x >= 0, 1, 0)            # return the result

    def fit(self, X, y):
        X = np.array(X, dtype=float)             # ensure numpy array
        y = np.array(y)
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.step_funct(linear_output)
                update = self.lr * (y[idx] - y_predicted)   # parentheses, not brackets
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        X = np.array(X, dtype=float)
        linear_output = np.dot(X, self.weights) + self.bias
        return self.step_funct(linear_output)


if __name__ == "__main__":
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]         # all rows length 2
    y = [0, 1, 1, 1]                             # this is OR logic

    model = Perceptron(lr=0.01, n_iters=10)
    model.fit(X, y)

    print("Learned Weights:", model.weights)
    print("Learned Bias:   ", model.bias)
    print("Predictions:    ", model.predict(X))
    
# OR = 0, 1, 1, 1
# AND = 0, 0, 0, 1 
# XOR = 0, 1, 1, 0 (motivates MLPs)
