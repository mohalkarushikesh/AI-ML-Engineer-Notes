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
