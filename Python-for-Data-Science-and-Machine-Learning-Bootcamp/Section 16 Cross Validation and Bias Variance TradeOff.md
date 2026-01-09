# 📊 Bias–Variance Trade‑Off

## 🧩 Definition
The **Bias–Variance Trade‑Off** describes the balance between a model’s ability to **fit training data** (bias) and its ability to **generalize to unseen data** (variance).  
- **Bias**: Error due to overly simplistic assumptions in the model.  
- **Variance**: Error due to sensitivity to small fluctuations in training data.  
- The trade‑off arises because reducing bias often increases variance, and reducing variance often increases bias.

---

## 🔑 Key Concepts

### 1. Bias
- **High Bias** → Model is too simple, underfits data.  
- Predictions are consistently off target.  
- Example: Linear regression applied to a highly nonlinear dataset.

### 2. Variance
- **High Variance** → Model is too complex, overfits data.  
- Predictions vary wildly depending on training set.  
- Example: Deep decision tree memorizing training data noise.

### 3. Training Error Behavior
- Training error decreases as model complexity increases.  
- High bias → high training error.  
- Low bias → low training error.

### 4. Test Error Behavior
- Test error initially decreases with complexity (better fit).  
- After a point, test error increases due to overfitting (high variance).  
- The curve is typically **U‑shaped**.

### 5. Optimal Model
- Achieves **low bias** and **low variance** simultaneously.  
- Found at the “sweet spot” of complexity where test error is minimized.  
- Balances accuracy and generalization.

---

## 📉 Mathematical View
Expected prediction error can be decomposed as:

$$
E[(y - \hat{f}(x))^2] = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}
$$

- **Bias²** → Error from incorrect assumptions.  
- **Variance** → Error from sensitivity to training data.  
- **Irreducible Error** → Noise inherent in the data (cannot be eliminated).

---

## 🧪 Examples
- **Linear Regression**: Low variance, high bias if data is nonlinear.  
- **Decision Trees**: Low bias, high variance if tree is deep.  
- **Random Forests**: Reduce variance by averaging multiple trees.  
- **Regularization (L1/L2)**: Adds bias to reduce variance, improving generalization.

---

## ⚖️ Strategies to Manage Trade‑Off
- **Cross‑Validation** → Estimate test error to find optimal complexity.  
- **Regularization** → Penalize complexity (Ridge, Lasso).  
- **Ensemble Methods** → Bagging/Boosting reduce variance.  
- **Feature Selection** → Remove irrelevant features to reduce variance.  
- **Early Stopping** → Prevent overfitting in iterative training.

---

## 📝 Cheatsheet Summary
- **High Bias** → Underfitting → Simple model.  
- **High Variance** → Overfitting → Complex model.  
- **Goal** → Minimize test error by balancing bias and variance.  
- **Equation** → Error = Bias² + Variance + Noise.  

---

✅ **In short:** The Bias–Variance Trade‑Off is about finding the “sweet spot” of model complexity where both bias and variance are low, ensuring the model generalizes well to unseen data.

---
