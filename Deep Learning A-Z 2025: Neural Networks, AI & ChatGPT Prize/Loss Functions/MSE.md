Alright, let’s dive into **Mean Squared Error (MSE)** in detail, as a standalone concept.

---

## 📌 What is MSE?
- **Mean Squared Error (MSE)** is a statistical measure used to evaluate how well a model’s predictions match the actual values.  
- It quantifies the **average squared difference** between predicted values and true values.  

Mathematically:
\[
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
\]

Where:
- \(n\) = number of data points  
- \(y_i\) = actual (true) value  
- \(\hat{y}_i\) = predicted value  

---

## 📌 Why Square the Errors?
- **Errors** are differences: \(y_i - \hat{y}_i\).  
- If we just summed errors, positive and negative values could cancel each other out.  
- Squaring ensures:
  - All errors are positive.  
  - Larger errors are penalized more heavily (quadratic growth).  

---

## 📌 Properties of MSE
- **Always non-negative**: Because squares are never negative.  
- **Sensitive to outliers**: A single large error increases MSE significantly.  
- **Units**: The result is in the square of the original unit (e.g., if values are in meters, MSE is in meters²).  

---

## 📌 Interpretation
- **Lower MSE** → better model fit (predictions are closer to actual values).  
- **Higher MSE** → worse fit (predictions deviate more).  
- MSE = 0 means **perfect predictions**.  

---

## 📌 Example
Suppose actual values are:
\[
y = [3, 5, 2]
\]
Predicted values are:
\[
\hat{y} = [2.5, 5.5, 2]
\]

Step 1: Errors
\[
[3-2.5, \; 5-5.5, \; 2-2] = [0.5, -0.5, 0]
\]

Step 2: Squared Errors
\[
[0.25, \; 0.25, \; 0]
\]

Step 3: Mean
\[
MSE = \frac{0.25 + 0.25 + 0}{3} = \frac{0.5}{3} \approx 0.167
\]

So the average squared error is **0.167**.

---

## 📌 Comparison with Other Metrics
- **MAE (Mean Absolute Error):** Uses absolute values instead of squares, less sensitive to outliers.  
- **RMSE (Root Mean Squared Error):** Square root of MSE, brings units back to original scale.  

---

👉 In short: **MSE is the average of squared prediction errors, a core metric for evaluating accuracy in regression and forecasting.**
