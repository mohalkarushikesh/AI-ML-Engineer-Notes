# 📘 Core Calculus Concepts in ML

## 🔹 Differentials
- **Definition:** Measure the *rate of change* of a function.  
- **Optimization:**  
  - Find **minima/maxima** of curves → critical points where derivative = 0.  
- **Applications:**  
  - **Engineering:** Max strength, efficiency.  
  - **Finance:** Minimize cost, maximize profit.  
  - **Machine Learning / Deep Learning:**  
    - **Gradient Descent** → minimize loss/cost function.  
    - **Gradient Ascent** → maximize reward (reinforcement learning).  
    - **Higher-order derivatives** → used in advanced optimizers (e.g., Newton’s method, Adam with momentum).  

---

## 🔹 Integrals
- **Definition:** Measure the *accumulated area under a curve*.  
- **Applications:**  
  - **Receiver Operating Characteristics (ROC):**  
    - Area under ROC curve (AUC) → performance metric in classification.  
  - **Probability Theory:**  
    - Expectation of random variables → $\mathbb{E}[X] = \int x \cdot p(x) \, dx$.  
    - Widely used in ML/DL for probabilistic models, Bayesian inference.  

---

## 🔹 Calculating Limits
- **Definition:** Describe the behavior of a function as input approaches a value.  
- **Role in Calculus:**  
  - Foundation for **derivatives** (rate of change).  
  - Foundation for **integrals** (area under curve).  
- **Applications:**  
  - Handling indeterminate forms ($\frac{0}{0}$, $\infty - \infty$).  
  - Ensuring stability in ML algorithms (e.g., avoiding exploding/vanishing gradients).  
  - Defining continuity and convergence in optimization.  

---

## 🎯 Summary
- **Differentials** → optimization (min/max, gradient descent).  
- **Integrals** → accumulation (area, probability, ROC-AUC).  
- **Limits** → foundation of calculus, stability, convergence.  

---
