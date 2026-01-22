# 📘 Session 9: Partial Derivative Calculus

## 1️⃣ Intro to Calculus
- **Calculus** is the study of **change**.  
- Two main branches:  
  - **Differential Calculus** → concerned with derivatives (slopes, rates of change).  
  - **Integral Calculus** → concerned with integrals (areas, accumulation).  
- **Derivative (single variable):**  
  - $f'(x) = \lim_{\Delta x \to 0} \frac{f(x+\Delta x) - f(x)}{\Delta x}$  
- **Interpretation:** Derivative gives the slope of the curve at a point.

---

## 2️⃣ Gradients in Machine Learning
- **Gradient** → generalization of derivative to multiple variables.  
- **Single-variable function:** Gradient = derivative.  
- **Multi-variable function:** Gradient = vector of partial derivatives.  
  - $\nabla f(x, y) = \left( \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y} \right)$  
- **Role in ML:**  
  - Gradients tell us how to adjust parameters to minimize loss.  
  - Optimizers (SGD, Adam) use gradients to update weights.  
- **Example:** For loss function $f(x,y)$, gradient points in the direction of steepest ascent; optimization moves in the opposite direction (descent).

---

## 3️⃣ Integrals
- **Integral (single variable):**  
  - $\int f(x) \, dx$  
  Represents the **area under the curve** or accumulated quantity.  
- **Definite Integral:**  
  - $\int_a^b f(x) \, dx$  
  → area between $x=a$ and $x=b$.  
- **Indefinite Integral:**  
  - General antiderivative, includes constant $C$.  
- **Applications in ML:**  
  - Probability distributions (area under PDF = 1).  
  - Expected values and continuous loss functions.  

---

## 4️⃣ Partial Derivatives
- **Definition:** Derivative of a multivariable function with respect to one variable, keeping others constant.  
  - $\frac{\partial f}{\partial x}, \quad \frac{\partial f}{\partial y}$  
- **Example:**  
  If $f(x,y) = x^2y + 3y$, then  
  - $\frac{\partial f}{\partial x} = 2xy$  
  - $\frac{\partial f}{\partial y} = x^2 + 3$  
- **Applications:**  
  - Gradients in ML (loss functions with many parameters).  
  - Optimization in high-dimensional spaces.  

---

## 🎯 Summary
- **Derivatives** → rate of change (slope).  
- **Gradients** → vector of partial derivatives (multi-variable).  
- **Integrals** → accumulation/area under curve.  
- **Partial Derivatives** → essential for ML optimization, backpropagation, and multivariable calculus.

---

<img width="1110" height="579" alt="image" src="https://github.com/user-attachments/assets/32afdcc5-3b0b-4702-b2f2-421c2ae5dab3" />

**athematical breakdown of regression for machine learning**. Let’s go point by point, starting from the basics and building up to the optimization process.

---

## 📌 1. Problem Setup
- We have **data points**: \((x_i, y_i)\) for \(i = 1, 2, \dots, n\).
- Goal: Find a function \(f(x)\) that predicts \(y\) from \(x\).
- In **linear regression**, we assume:
  \[
  f(x) = w \cdot x + b
  \]
  where \(w\) is the weight (slope) and \(b\) is the bias (intercept).

---

## 📌 2. Hypothesis Function
- For multiple features (\(x \in \mathbb{R}^d\)):
  \[
  \hat{y}_i = w_1x_{i1} + w_2x_{i2} + \dots + w_dx_{id} + b
  \]
- In vector form:
  \[
  \hat{y}_i = \mathbf{w}^T \mathbf{x}_i + b
  \]

---

## 📌 3. Loss Function
- We measure error using **Mean Squared Error (MSE)**:
  \[
  L(w, b) = \frac{1}{n} \sum_{i=1}^n \big(\hat{y}_i - y_i\big)^2
  \]
- This penalizes large deviations between prediction and actual values.

---

## 📌 4. Optimization (Gradient Descent)
- We want to minimize \(L(w, b)\).
- Compute partial derivatives:
  - Gradient w.r.t. weight:
    \[
    \frac{\partial L}{\partial w_j} = \frac{2}{n} \sum_{i=1}^n \big(\hat{y}_i - y_i\big) x_{ij}
    \]
  - Gradient w.r.t. bias:
    \[
    \frac{\partial L}{\partial b} = \frac{2}{n} \sum_{i=1}^n \big(\hat{y}_i - y_i\big)
    \]
- Update rules:
  \[
  w_j \leftarrow w_j - \alpha \frac{\partial L}{\partial w_j}, \quad b \leftarrow b - \alpha \frac{\partial L}{\partial b}
  \]
  where \(\alpha\) is the learning rate.

---

## 📌 5. Closed‑Form Solution (Normal Equation)
- Instead of iterative updates, linear regression has an exact solution:
  \[
  \mathbf{w} = (X^TX)^{-1}X^Ty
  \]
  where:
  - \(X\) is the design matrix (\(n \times d\)),
  - \(y\) is the vector of outputs.

---

## 📌 6. Regularization (to prevent overfitting)
- **Ridge Regression (L2 penalty):**
  \[
  L(w, b) = \frac{1}{n} \sum_{i=1}^n (\hat{y}_i - y_i)^2 + \lambda \|\mathbf{w}\|^2
  \]
- **Lasso Regression (L1 penalty):**
  \[
  L(w, b) = \frac{1}{n} \sum_{i=1}^n (\hat{y}_i - y_i)^2 + \lambda \|\mathbf{w}\|_1
  \]

---

## 📌 7. Evaluation Metrics
- Beyond MSE, we often use:
  - **R² (coefficient of determination):**
    \[
    R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
    \]
  - **MAE (Mean Absolute Error):**
    \[
    MAE = \frac{1}{n} \sum |y_i - \hat{y}_i|
    \]

---

✅ That’s the mathematical backbone of regression in ML:  
- Define hypothesis → choose loss → optimize (gradient descent or closed form) → regularize → evaluate.

---

Perfect! Let’s walk through a **numerical example of linear regression step by step** so you can see the math in action.

## 📊 Example Dataset
We’ll fit a line to 3 points:

\[
(1, 2), \quad (2, 3), \quad (3, 4)
\]

So \(x = [1, 2, 3]\) and \(y = [2, 3, 4]\).

---

## 1. Hypothesis Function
We assume:
\[
\hat{y} = w \cdot x + b
\]

---

## 2. Loss Function (MSE)
\[
L(w, b) = \frac{1}{n} \sum_{i=1}^n (\hat{y}_i - y_i)^2
\]

Here \(n = 3\).

---

## 3. Closed‑Form Solution (Normal Equation)
For simple linear regression, the slope and intercept can be computed directly:

- **Slope \(w\):**
\[
w = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}
\]

- **Intercept \(b\):**
\[
b = \bar{y} - w \cdot \bar{x}
\]

---

## 4. Compute Means
\[
\bar{x} = \frac{1+2+3}{3} = 2, \quad \bar{y} = \frac{2+3+4}{3} = 3
\]

---

## 5. Compute Slope
\[
w = \frac{(1-2)(2-3) + (2-2)(3-3) + (3-2)(4-3)}{(1-2)^2 + (2-2)^2 + (3-2)^2}
\]

\[
w = \frac{(-1)(-1) + (0)(0) + (1)(1)}{(-1)^2 + 0^2 + 1^2}
\]

\[
w = \frac{1 + 0 + 1}{1 + 0 + 1} = \frac{2}{2} = 1
\]

---

## 6. Compute Intercept
\[
b = \bar{y} - w \cdot \bar{x} = 3 - (1)(2) = 1
\]

---

## ✅ Final Regression Line
\[
\hat{y} = x + 1
\]

---

## 7. Predictions
- For \(x=1\): \(\hat{y} = 2\)  
- For \(x=2\): \(\hat{y} = 3\)  
- For \(x=3\): \(\hat{y} = 4\)  

It fits the points **perfectly**.

---

So in this toy example, the regression line is exactly the line through the points: slope \(=1\), intercept \(=1\).

---


Here’s the plot of the regression line 
𝑦
=
𝑥
+
1
 with the sample points (1,2), (2,3), and (3,4). The red line is the fitted regression line, and the blue dots are the data points.

You can see that the line passes exactly through all three points, which means the regression fit is perfect in this case.

🔎 What This Shows
Data points (blue): The actual observed values.

Regression line (red): The predicted relationship between 
𝑥
 and 
𝑦
.

Since the points lie exactly on the line, the error (MSE) is zero.

This is a rare case in real-world data, but it’s a great illustration of how regression works mathematically and visually.

Would you like me to extend this example to a non-perfect dataset (say, points like (1,2), (2,2.5), (3,4)) so you can see how regression minimizes error when the line doesn’t fit perfectly?

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/356afaa6-66b4-43d9-8ae6-58f9d4d376d0" />
