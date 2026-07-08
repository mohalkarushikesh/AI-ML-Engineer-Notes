# Perceptron from scratch — Implement a single neuron with NumPy to solve a linearly separable problem (e.g., AND/OR gates); code the forward pass and weight updates by hand.

"""
A single-layer perceptron implemented from scratch with NumPy.

The perceptron is the simplest neural network: one layer of weights, no hidden
units, and a hard step activation. It is a BINARY LINEAR CLASSIFIER — it learns
a straight line (hyperplane) that separates two classes.

Pipeline:
    1. Initialize weights and bias
    2. Compute the weighted sum of the input features  (linear combination)
    3. Pass that sum through a step activation function  -> predicted class (0/1)
    4. Nudge the parameters whenever a sample is misclassified

Note: it only converges if the data is LINEARLY SEPARABLE (e.g. OR, AND).
XOR is NOT linearly separable and cannot be solved by a single perceptron.
"""

import numpy as np


class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        # learning_rate (lr): step size for each weight update.
        # n_iters: number of full passes (epochs) over the training data.
        self.lr = learning_rate
        self.n_iters = n_iters
        # Weights/bias are unknown until fit() is called, so start as None.
        self.weights = None
        self.bias = None

    def _unit_step_func(self, x):
        # Activation function (Heaviside step): returns 1 if x >= 0, else 0.
        # This is what makes the output a discrete class label rather than a
        # continuous value. It is non-differentiable, which is why the
        # perceptron uses an error-driven update rule instead of backprop.
        return np.where(x >= 0, 1, 0)

    def fit(self, X, y):
        # X: feature matrix of shape (n_samples, n_features)
        # y: target labels of shape (n_samples,), each 0 or 1
        n_features = X.shape[1]

        # 1. Initialize the weights and bias to zero.
        #    (Unlike deeper networks, zero init is fine here — there is only
        #     one neuron, so there is no symmetry to break.)
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        # 2. Iteratively update parameters over epochs.
        for _ in range(self.n_iters):
            # Go through the training samples one at a time (online learning).
            for idx, x_i in enumerate(X):

                # Linear combination: w . x + b
                linear_output = np.dot(x_i, self.weights) + self.bias

                # Squash to a class label via the step function.
                y_predicted = self._unit_step_func(linear_output)

                # Perceptron learning rule:
                #   error = (y_true - y_pred) -> one of {-1, 0, +1}
                #   If the prediction is correct, error = 0 -> no change.
                #   Otherwise, push the weights toward the correct answer.
                update = self.lr * (y[idx] - y_predicted)
                self.weights += update * x_i   # w += lr * error * x
                self.bias += update            # b += lr * error

    def predict(self, X):
        # Inference: same linear combination + step function, no updates.
        # Works on a single sample or a whole batch thanks to np.dot broadcasting.
        linear_output = np.dot(X, self.weights) + self.bias
        return self._unit_step_func(linear_output)


if __name__ == "__main__":
    # Toy dataset: OR gate (linearly separable, so the perceptron can learn it).
    # Each row is one sample [x1, x2]; y is the expected OR output.
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 1])

    # Initialize and train.
    clf = Perceptron(learning_rate=0.1, n_iters=10)
    clf.fit(X, y)

    # Output the learned parameters and predictions.
    print("Learned Weights:", clf.weights)
    print("Learned Bias:", clf.bias)
    print("Predictions:", clf.predict(X))


# =============================================================================
# NOTES
# =============================================================================
# Example run:
#   Learned Weights: [0.1 0.1]
#   Learned Bias:    -0.1
#   Predictions:     [0 1 1 1]   -> matches y, so the OR gate is learned
#
# How the learned parameters classify each input:
#   decision boundary -> 0.1*x1 + 0.1*x2 - 0.1 >= 0
#   [0,0]: -0.1  < 0  -> 0
#   [0,1]:  0.0 >= 0  -> 1
#   [1,0]:  0.0 >= 0  -> 1
#   [1,1]:  0.1 >= 0  -> 1
#
# Key ideas:
#   - Converges only when the classes are LINEARLY SEPARABLE.
#     OR and AND work; XOR does NOT (needs a hidden layer / MLP).
#   - Update rule: w += lr * (y_true - y_pred) * x,  b += lr * (y_true - y_pred).
#     Correct prediction -> error 0 -> no change.
#   - The step activation is non-differentiable, hence the error-driven rule
#     instead of gradient descent / backprop.
#   - learning_rate only scales the step size; for a plain perceptron it does
#     not change WHETHER it converges, only how fast the weights grow.
#
# Try next:
#   - Swap y to the AND gate ([0,0,0,1]) — still learnable.
#   - Swap y to XOR ([0,1,1,0]) — watch it fail to converge (motivates MLPs).
