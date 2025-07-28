#### **🔹 Linear Regression**
Linear Regression is a **fundamental statistical and machine learning method** used to model relationships between variables. It predicts a **continuous dependent variable** based on one or more independent variables, assuming a **linear relationship**.

---

## **🔸 Historical Context**
- The term **"Regression"** was coined by **Sir Francis Galton** in the 19th century.
- He observed that the heights of sons tended to **regress towards** the population's average height instead of mirroring their fathers' extreme heights.  
- This concept inspired the name **"Regression"**, which describes the tendency of dependent variables to settle around a central trend.

---

## **🔸 Core Concept: Linear Relationship**
Linear Regression models the relationship between **dependent variable ($Y$)** and **independent variable(s) ($X$)** using a straight-line approach.

### **📌 Equation of Linear Regression**

$$
Y = \beta_0 + \beta_1 X + \epsilon
$$

Where:
- $Y$ → Dependent variable  
- $X$ → Independent variable  
- $\beta_0$ → Intercept (value of $Y$ when $X = 0$)  
- $\beta_1$ → Slope (rate of change in $Y$ per unit change in $X$)  
- $\epsilon$ → Error term (accounts for randomness in data)

### **📌 Key Methodology**
- The **Least Squares Method** is used to **find the best-fit line** by **minimizing the sum of squared residuals**.
- Residual (error) = Actual value - Predicted value

$$
\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$

---

## **🔸 Types of Linear Regression**

### **1️⃣ Simple Linear Regression (SLR)**
- **Uses**: Models the relationship between **one independent variable ($X$)** and **one dependent variable ($Y$)**.
- **Equation**:

$$
Y = \theta_0 + \theta_1 X + \epsilon
$$

- **Example**: Predicting **salary** based on **years of experience**.

### **2️⃣ Multiple Linear Regression (MLR)**
- **Uses**: Models relationships between **multiple independent variables ($X_1, X_2, ..., X_n$)** and **one dependent variable ($Y$)**.
- **Equation**:

$$
Y = \theta_0 + \theta_1 X_1 + \theta_2 X_2 + \dots + \theta_n X_n + \epsilon
$$

- **Example**: Predicting **house prices** based on **square footage, number of bedrooms, location, and amenities**.

### **3️⃣ Polynomial Regression**
- **Uses**: Captures **non-linear relationships** by introducing polynomial terms.
- **Equation**:

$$
Y = \theta_0 + \theta_1 X + \theta_2 X^2 + \theta_3 X^3 + \dots + \theta_n X^n + \epsilon
$$

- **Example**: Predicting **growth rate** of bacteria over time when the trend isn’t linear.

### **4️⃣ Ridge Regression (L2 Regularization)**
- **Uses**: Handles **multicollinearity** by **penalizing large coefficients**.
- **Equation**:

$$
\min \sum_{i=1}^{n} (y_i - y_i^*)^2 + \lambda \sum_{j=1}^{p} \theta_j^2
$$

- **Example**: Predicting **stock prices** with correlated indicators.

### **5️⃣ Lasso Regression (L1 Regularization)**
- **Uses**: Shrinks some coefficients to **zero**, enabling **feature selection**.
- **Equation**:

$$
\min \sum_{i=1}^{n} (y_i - y_i^*)^2 + \lambda \sum_{j=1}^{p} |\theta_j|
$$

- **Example**: Selecting **key factors** affecting **customer satisfaction**.

### **6️⃣ Elastic Net Regression**
- **Uses**: Combines **L1 and L2 regularization** to balance selection and shrinkage.
- **Equation**:

$$
\min \sum_{i=1}^{n} (y_i - y_i^*)^2 + \lambda_1 \sum_{j=1}^{p} |\theta_j| + \lambda_2 \sum_{j=1}^{p} \theta_j^2
$$

- **Example**: Predicting **sales revenue** using interrelated factors.

### **7️⃣ Stepwise Regression**
- **Uses**: **Automates feature selection** by adding/removing predictors based on significance.
- **Example**: **Medical diagnosis models** using relevant patient data.

### **8️⃣ Quantile Regression**
- **Uses**: Predicts **percentiles** of dependent variable distribution instead of the mean.
- **Example**: **Predicting house prices in different market segments**.

### **9️⃣ Bayesian Regression**
- **Uses**: Estimates parameters using **probability distributions**, ideal for **small or uncertain data**.
- **Example**: **Weather forecasting with incomplete data**.

---

## **🔸 Assumptions of Linear Regression**
✔ **Linear Relationship** → $X$ and $Y$ must have a linear correlation  
✔ **Normal Distribution of Residuals** → Residuals should follow normal distribution  
✔ **Homoscedasticity** → Constant variance of residuals across $X$ values  
✔ **Minimal Multicollinearity** → Low correlation among predictors in MLR

---

## **🔸 Evaluation Metrics**

### **📌 Mean Absolute Error (MAE)**
$$
MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$
- Average absolute difference between actual and predicted values  
- Lower $MAE$ indicates better accuracy

### **📌 Mean Squared Error (MSE)**
$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$
- Penalizes larger errors; sensitive to outliers

### **📌 Root Mean Squared Error (RMSE)**
$$
RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$
- Expressed in units of $Y$ for better interpretability

---

## **🔸 Practical Applications**
Linear Regression is widely used in:
- 📊 **Sales Forecasting**  
- 💡 **Trend Analysis**  
- 📈 **Stock Market Predictions**  
- 🏠 **Real Estate Pricing**  
- 🏥 **Medical Research**

---

## **🔸 Common Python Libraries for Regression**
```python
import pandas as pd    # Data Manipulation
import numpy as np     # Numerical Computations
import matplotlib.pyplot as plt    # Visualizations
import seaborn as sns  # Advanced Visualizations
```

### **📌 Example Visualization**

#### **Pairplot (Correlation Visualization)**
```python
sns.pairplot(df)
plt.show()
```

#### **Distribution Plot**
```python
sns.histplot(df['ColumnName'])
plt.show()
```

---

## **🔸 Final Thoughts**
Linear Regression is a **simple yet powerful tool** to **model relationships, uncover trends, and predict outcomes**. From simple lines to regularized magic like **Lasso** or **Elastic Net**, choosing the **right variant** ensures **accuracy, interpretability, and efficiency**.

---
