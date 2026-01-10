**In regression analysis, the most common evaluation metrics are MSE, RMSE, MAE, and R². MSE and RMSE measure squared error magnitude, MAE captures average absolute error, and R² explains variance explained by the model. Together, they provide a balanced view of accuracy, robustness, and explanatory power.**

---

# 📊 Regression Evaluation Metrics

## 🔹 Mean Squared Error (MSE)
- **Definition**: Average of squared differences between predicted and actual values.  
- **Formula**:  
  $MSE = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$  
- **Interpretation**: Penalizes larger errors more heavily due to squaring.  
- **Strengths**: Sensitive to large deviations, useful when big errors are critical.  
- **Limitations**: Units are squared, making interpretation less intuitive.  

---

## 🔹 Root Mean Squared Error (RMSE)
- **Definition**: Square root of MSE, bringing error back to original units.  
- **Formula**:  
  $RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$  
- **Interpretation**: Represents the average magnitude of error in the same units as the target variable.  
- **Strengths**: Easier to interpret than MSE, widely used in practice.  
- **Limitations**: Still penalizes large errors more than small ones.  

---

## 🔹 Mean Absolute Error (MAE)
- **Definition**: Average of absolute differences between predicted and actual values.  
- **Formula**:  
  $MAE = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$  
- **Interpretation**: Represents average error magnitude without squaring.  
- **Strengths**: More robust to outliers compared to MSE/RMSE.  
- **Limitations**: Does not penalize large errors as strongly, may understate their impact.  

---

## 🔹 Coefficient of Determination (R²)
- **Definition**: Proportion of variance in the dependent variable explained by the model.  
- **Formula**:  
  $R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$  
- **Interpretation**:  
  - $R^2 = 1$: Perfect fit.  
  - $R^2 = 0$: Model explains no variance.  
  - Negative $R^2$: Model performs worse than predicting the mean.  
- **Strengths**: Intuitive measure of explanatory power.  
- **Limitations**: Can be misleading in non-linear models or with overfitting.  

---

## 📝 Comparison Table

| Metric | Formula | Units | Sensitivity | Best Use Case |
|--------|---------|-------|-------------|---------------|
| **MSE** | $\frac{1}{n}\sum (y_i - \hat{y}_i)^2$ | Squared | High (large errors penalized) | When large errors are critical |
| **RMSE** | $\sqrt{\frac{1}{n}\sum (y_i - \hat{y}_i)^2}$ | Same as target | High | General regression evaluation |
| **MAE** | $\frac{1}{n}\sum |y_i - \hat{y}_i|$ | Same as target | Moderate | Robust to outliers |
| **R²** | $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$ | Dimensionless | N/A | Variance explained by model |

---

## ⚠️ Key Insights
- **MSE/RMSE**: Good for emphasizing large errors.  
- **MAE**: Better for datasets with outliers.  
- **R²**: Explains variance but does not measure error magnitude.  
- **Best practice**: Use multiple metrics together for a holistic evaluation.  

---

Sources: [GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/regression-metrics/), [Joyoshish Saha Blog](https://joyoshish.github.io/blog/2025/ds010-heartofml-regression4/), [CodePointTech](https://codepointtech.com/mastering-regression-metrics-r%c2%b2-mse-mae-explained-2/)

---
