import numpy as np
import matplotlib.pyplot as plt

# 1. THE ADVANCED MODEL CLASS
class AdvancedLinearRegression:
    """
    Advanced Linear Regression using Gradient Descent and L2 (Ridge) Regularization.
    It solves the equation: y = Xw + b
    """
    def __init__(self, learning_rate=0.01, n_iters=1000, l2_penalty=0.01):
        self.lr = learning_rate        # α: Controls the size of the step during gradient descent
        self.n_iters = n_iters        # How many times to loop through the data
        self.lambda_reg = l2_penalty   # λ: Penalty term to keep weights small (prevents overfitting)
        self.weights = None            # w: Slopes for each feature
        self.bias = None               # b: The y-intercept
        self.loss_history = []         # List to store the 'cost' at each iteration for plotting

    def fit(self, X, y):
        """
        The training process: Iteratively optimizes weights and bias to minimize Mean Squared Error.
        """
        # n_samples (m): Rows in dataset | n_features (n): Columns (variables) in dataset
        n_samples, n_features = X.shape
        
        # Initialize parameters: Start with 0s and let the math 'find' the correct values
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iters):
            # --- 1. THE FORWARD PASS (PREDICTION) ---
            # Dot product: Multiply each feature by its weight and add the bias
            # y_pred = (w1*x1 + w2*x2 ... + wn*xn) + b
            y_pred = np.dot(X, self.weights) + self.bias
            
            # --- 2. CALCULATE LOSS (COST) ---
            # MSE: Average of (Actual - Predicted)^2
            mse = np.mean((y_pred - y)**2)
            
            # Ridge Regularization: Adds a penalty proportional to the square of weights
            # This 'shrinks' the weights to ensure the model isn't too complex.
            reg_loss = self.lambda_reg * np.sum(self.weights**2)
            
            # Total Loss = Error + Complexity Penalty
            self.loss_history.append(mse + reg_loss)

            # --- 3. THE BACKWARD PASS (GRADIENTS) ---
            # Gradient of MSE: 2/m * X^T * (Error)
            # This tells us the direction of the steepest 'uphill' slope
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y)) + (2 * self.lambda_reg * self.weights)
            
            # Gradient of Bias: Simple average of the error
            db = (1 / n_samples) * np.sum(y_pred - y)

            # --- 4. UPDATE PARAMETERS (OPTIMIZATION) ---
            # We subtract the gradient because we want to move 'downhill' toward zero error
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        """
        Uses the learned weights and bias to predict y for new X values.
        """
        return np.dot(X, self.weights) + self.bias

    def score(self, X, y):
        """
        Calculates the R-squared (Coefficient of Determination) score.
        1.0 = Perfect prediction | 0.0 = Predicting the average every time.
        """
        y_pred = self.predict(X)
        # Sum of Squared Residuals (Error we still have)
        ss_res = np.sum((y - y_pred)**2)
        # Total Sum of Squares (Variance in the original data)
        ss_tot = np.sum((y - np.mean(y))**2)
        # Formula for R^2
        return 1 - (ss_res / ss_tot)

# 2. GENERATE SYNTHETIC DATA
# Setting seed ensures you get the same 'random' dots every time you run it
np.random.seed(42)
X = 2 * np.random.rand(100, 1) # Generate 100 values between 0 and 2
# Create a linear relationship: y = 3x + 4, then add 'noise' so dots don't form a perfect line
y = 4 + 3 * X.squeeze() + np.random.randn(100) * 0.5 

# 3. TRAIN THE MODEL
# High learning rate (0.1) works here because the data is simple and scaled
model = AdvancedLinearRegression(learning_rate=0.1, n_iters=200, l2_penalty=0.01)
model.fit(X, y)

# 4. PRINT RESULTS
print(f"Final Weights: {model.weights}") # Should be near 3.0
print(f"Final Bias: {model.bias:.4f}")   # Should be near 4.0
print(f"R-squared Score: {model.score(X, y):.4f}")

# 5. VISUALIZATION
plt.figure(figsize=(12, 5))

# Plot 1: The Learning Process
# If this curve is going down, the model is successfully 'learning'
plt.subplot(1, 2, 1)
plt.plot(model.loss_history, color='green', linewidth=2)
plt.title("Learning Curve (Loss decreasing)")
plt.xlabel("Iterations (Steps)")
plt.ylabel("Cost (Error + Penalty)")

# Plot 2: The result
# Shows how the 'Best Fit Line' actually slices through the data
plt.subplot(1, 2, 2)
plt.scatter(X, y, color='blue', alpha=0.5, label='Actual Data')
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Prediction Line')
plt.title("Advanced Linear Fit")
plt.legend()

plt.tight_layout()
plt.show()
