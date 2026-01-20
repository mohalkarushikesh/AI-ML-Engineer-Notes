# 🧠 Technical Deep Dive: The Mathematics of Linear Regression

This project implements a **Multivariate Linear Regression** model using **Batch Gradient Descent** and **Tikhonov Regularization (L2)**. Below is the step-by-step mathematical breakdown of the logic executed in the `fit()` and `predict()` methods.

---

## 1. The Hypothesis Function (Forward Pass)

The model assumes a linear relationship between the input features and the target variable. In vectorized form, the prediction for a batch of data is calculated as:

* ****: An  matrix where  is the number of samples and  is the number of features.
* ****: A vector of weights (parameters) for each feature.
* ****: The bias scalar (intercept).
* **Vectorization:** Instead of looping through  samples, we use a single dot product (`np.dot(X, self.weights)`), which leverages modern CPU SIMD instructions for high-performance computation.

---

## 2. The Regularized Cost Function

To evaluate how well the model is performing, we use **Mean Squared Error (MSE)**. However, to prevent **Overfitting** (where the model assigns extreme weights to fit noise), we add an **L2 Penalty** (Ridge Regression).

The total objective function  is:

* **MSE Term:** Penalizes the distance between predictions and actual values.
* **L2 Term ():** Penalizes the magnitude of the weights. A higher  forces the weights to be smaller, resulting in a "simpler" model that generalizes better to unseen data.

---

## 3. Optimization via Gradient Descent

To minimize the cost function, we calculate the partial derivatives (gradients) of  with respect to our parameters  and .

### Partial Derivative for Weights ():

### Partial Derivative for Bias ():

**Note on the Transpose ():** The transpose operation in `np.dot(X.T, (y_pred - y))` is mathematically necessary to align the feature dimensions with the error residuals, effectively calculating the "responsibility" of each feature for the total error.

### The Update Rule:

We move the parameters in the **opposite** direction of the gradient to find the local minimum:



*(Where  is the Learning Rate)*.

---

## 4. Evaluation Metric: Coefficient of Determination ()

To quantify the model's predictive power, we implement the  score:

* ** (Residual Sum of Squares):**  — The error our model makes.
* ** (Total Sum of Squares):**  — The total variance in the data.

An  of **1.0** indicates the model explains all the variability of the response data around its mean, while a score of **0.0** indicates the model performs no better than simply predicting the mean of  for every input.

---

## 5. Implementation Details

* **Parameter Initialization:** Weights are initialized to zero (`np.zeros`). While Neural Networks require random initialization to break symmetry, Linear Regression is a convex optimization problem, so starting at zero always leads to the global minimum.
* **Broadcasting:** In the line `y_pred = np.dot(X, self.weights) + self.bias`, NumPy uses "broadcasting" to add the single scalar `bias` value to every element in the resulting prediction vector.

---

To truly understand how this model "thinks," we have to look at the geometry and the calculus that drives the learning process. Here is a deep dive into the technical implementation.

---

### 1. The Geometry of the Hypothesis

The model assumes that the target  can be represented as a linear combination of input features . In a 2D space, this is a line; in higher dimensions, it is a **hyperplane**.

In the code, `np.dot(X, self.weights)` performs a high-speed matrix-vector multiplication. If  has dimensions  and  is , the result is a  vector of predictions. The `+ self.bias` part uses **NumPy broadcasting** to add a single scalar to all 100 predictions simultaneously.

---

### 2. The Cost Function (The "Mountain")

We define a **Surface of Error** (the Cost Function). Our goal is to find the lowest point in this valley. We use **Mean Squared Error (MSE)** plus **L2 Regularization**.

* **Why square the error?** Squaring ensures that larger errors are penalized much more heavily than small ones, and it makes the function "convex" (bowl-shaped), which is easy to optimize.
* **Why the L2 Penalty?** It prevents any single weight from becoming too large. In high-dimensional data, this stops the model from "chasing noise" and over-fitting.

---

### 3. The Calculus: Gradient Descent

To find the bottom of the bowl, we take the **Partial Derivative** of the cost function with respect to  and . This tells us the "slope" of the error.

**The Weight Gradient ():**


**The Bias Gradient ():**


* **The Transpose ():** This is mathematically critical. Multiplying the transposed features by the error vector calculates the **correlation** between each feature and the prediction error.
* **The Update Step:** We move the parameters in the *opposite* direction of the gradient.



---

### 4. Evaluation: R-squared ()

The  score implemented in the `score()` method is the "Coefficient of Determination." It compares your model to a "dumb" model that simply predicts the average value of  every time.

* ****: The sum of the squares of your model's errors.
* ****: The total variance in the actual data.
* **Meaning**: If , your model explains 90% of the variation in the data.

---

### 5. Summary of Technical Logic

1. **Initialization**: Start with .
2. **Forward Pass**: Calculate predictions.
3. **Compute Loss**: Measure how far off we are (plus the L2 penalty).
4. **Backward Pass**: Calculate the gradients (the slope of the error).
5. **Optimization**: Step down the slope using the learning rate.
6. **Repeat**: Do this for `n_iters` until the loss curve flattens.
