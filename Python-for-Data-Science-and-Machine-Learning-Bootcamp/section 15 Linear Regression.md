### **Linear Regression**

Linear regression is one of the simplest and most widely used statistical methods to understand relationships between variables.

#### **Historical Context**
- The term **"regression"** was coined by Sir Francis Galton in the 19th century.
- He observed that the heights of sons tended to regress, or drift towards, the average height (mean height) of the population, rather than being as extreme as their fathers'. This phenomenon inspired the term "regression."

#### **Classic Linear Regression**
Linear regression aims to model the relationship between a dependent variable (\(y\)) and one or more independent variables (\(x\)). It assumes a linear relationship between these variables.

- **Method**:
  - The **least squares method** is used to find the best-fit line.
  - This is done by **minimizing the sum of the squares of the residuals** (the differences between observed and predicted values). Mathematically:
    $$\text{Residual} = y_{\text{observed}} - y_{\text{predicted}}$$
    $$\text{Sum of Squares of Residuals} = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

- **Equation of the Line**:
  $$y = \beta_0 + \beta_1x + \epsilon$$
  Where:
  - \(y\): Dependent variable
  - \(x\): Independent variable
  - \(\beta_0\): Intercept of the line
  - \(\beta_1\): Slope of the line
  - \(\epsilon\): Error term (captures randomness or noise)

---

### **Linear Regression & Best-Fit Line**

- The **best-fit line** equation represents a straight-line relationship between the dependent variable (Y) and independent variable(s) (X). The slope of this line indicates how much **Y changes** for a **unit change in X**.

- **Linear Regression** is a predictive modeling technique used to estimate **Y (dependent variable)** based on **X (independent variable)**. Since it establishes a linear relationship between X and Y, the method is named **Linear Regression**.

- In the example above:
  - **X (input):** Work experience  
  - **Y (output):** Salary  
  - The regression line is the **best-fit line** that minimizes the error between predicted values and actual values.

---

### **Hypothesis Function in Linear Regression**
- The equation for linear regression is:  
  $$Y^ = \theta_1 + \theta_2 X$$  
  where:
  - **θ₁:** Intercept (bias term)
  - **θ₂:** Coefficient of X (slope)

---

### **Optimizing θ₁ and θ₂ for Best-Fit Line**
To find the best-fit line, we minimize the error between predictions and actual values using **Mean Squared Error (MSE)**:

$$\min \sum_{i=1}^{n} (y_i^* - y_i)^2$$  

This represents the sum of squared differences between **actual values (y_i)** and **predicted values (y_i*)**.

- **Gradient Descent** is commonly used to iteratively update θ₁ and θ₂ by minimizing this error.

---

### **Types of Linear Regression**

### **1. Simple Linear Regression (SLR)**
- **Purpose:** Models the relationship between a **single** independent variable (X) and a dependent variable (Y) using a **straight line**.
- **Equation:**  
  $$Y = \theta_0 + \theta_1 X + \epsilon$$  
  where:
  - \( \theta_0 \) is the **intercept**
  - \( \theta_1 \) is the **slope** (how much Y changes per unit of X)
  - \( \epsilon \) represents **random error**
- **Example:** Predicting a person's **salary** based on **years of experience**.

---

### **2. Multiple Linear Regression (MLR)**
- **Purpose:** Models the relationship between **multiple** independent variables (X₁, X₂, ..., Xₙ) and a **single** dependent variable (Y).
- **Equation:**  
  $$Y = \theta_0 + \theta_1 X_1 + \theta_2 X_2 + ... + \theta_n X_n + \epsilon$$  
- **Example:** Predicting house prices based on factors like **square footage, number of bedrooms, location, and amenities**.

---

### **3. Polynomial Regression**
- **Purpose:** Extends linear regression by introducing **higher-degree polynomial terms**, capturing **non-linear** relationships.
- **Equation:**  
  $$Y = \theta_0 + \theta_1 X + \theta_2 X^2 + \theta_3 X^3 + ... + \theta_n X^n + \epsilon$$  
- **Example:** Predicting **growth rate** of bacteria over time, where the pattern **doesn’t follow a straight line**.

---

### **4. Ridge Regression (L2 Regularization)**
- **Purpose:** Handles **multicollinearity** (high correlation between independent variables) by **penalizing large coefficients**, making the model more stable.
- **Equation:**  
  $$\min \sum_{i=1}^{n} (y_i - y_i^*)^2 + \lambda \sum_{j=1}^{p} \theta_j^2$$  
  where:
  - \( \lambda \) is the **regularization parameter** (controls penalty strength)
  - \( \sum_{j=1}^{p} \theta_j^2 \) prevents overfitting by shrinking coefficient values.
- **Example:** Predicting **stock prices** with multiple correlated financial indicators.

---

### **5. Lasso Regression (L1 Regularization)**
- **Purpose:** Similar to Ridge regression, but it **shrinks some coefficients to zero**, effectively performing **feature selection**.
- **Equation:**  
  $$\min \sum_{i=1}^{n} (y_i - y_i^*)^2 + \lambda \sum_{j=1}^{p} |\theta_j|$$  
- **Example:** Selecting **most impactful factors** affecting **customer satisfaction** in a survey analysis.

---

### **6. Elastic Net Regression**
- **Purpose:** Combines **Ridge (L2) and Lasso (L1) regularization**, balancing feature selection and coefficient shrinkage.
- **Equation:**  
  $$\min \sum_{i=1}^{n} (y_i - y_i^*)^2 + \lambda_1 \sum_{j=1}^{p} |\theta_j| + \lambda_2 \sum_{j=1}^{p} \theta_j^2$$  
- **Example:** Predicting **sales revenue** using numerous interrelated factors.

---

### **7. Logistic Regression (Used for Classification)**
- **Purpose:** Though named "regression," it is used for **binary classification problems** (predicting categories).
- **Equation:**  
  $$p(Y=1) = \frac{1}{1+e^{-(\theta_0 + \theta_1 X_1 + \theta_2 X_2 + ... + \theta_n X_n)}}$$  
- **Example:** Predicting **whether a customer will buy a product (Yes/No)** based on demographics and browsing behavior.

---

Would you like to dive deeper into **gradient descent optimization** or **real-world applications** of these regressions? 🚀

#### **Applications**
Linear regression is widely applied in:
- Predicting trends (e.g., sales forecasting)
- Determining relationships (e.g., the impact of advertising spend on sales)
- Estimating unknown variables (e.g., predicting a person's weight based on their height)

#### **Assumptions of Linear Regression**
1. The relationship between independent and dependent variables is linear.
2. Residuals (errors) are normally distributed.
3. Homoscedasticity: The variance of residuals is constant across all levels of the independent variable.
4. There is minimal multicollinearity (in the case of multiple independent variables).

---

### **Common Library Imports**
When starting a project in Python for data analysis or machine learning, the following libraries are essential:
```python
import pandas as pd    # For data manipulation and analysis
import numpy as np     # For numerical computations
import matplotlib.pyplot as plt    # For creating visualizations
import seaborn as sns  # For advanced visualizations
```

---

### **Visualization with Seaborn**
1. **`sns.pairplot()`**:
   - Visualizes pairwise relationships between all numerical columns in the DataFrame.
   - **Syntax**:
     ```python
     sns.pairplot(df)  # Pass the entire DataFrame
     plt.show()
     ```
   - **Use Case**:
     Ideal for exploring correlations and distributions between multiple features in a dataset.

2. **`sns.distplot()`**:
   - Plots the distribution (histogram + kernel density estimate) of a single variable.
   - **Syntax**:
     ```python
     sns.distplot(df['ColumnName'])  # Pass the column to visualize
     plt.show()
     ```
   - **Use Case**:
     Useful for examining the spread, skewness, and variability of a single feature.

---

### **Quick Documentation Shortcut**
Use `Shift + Tab` in Jupyter Notebook or any compatible IDE to access function documentation and parameter details.
- Example: Highlight the function `sns.pairplot` and press **Shift + Tab** to quickly view available parameters and their descriptions.

---

### **Regression Evaluation Metrics**
In regression problems, evaluating the model's performance is crucial. Below are the most common metrics used:

1. **Mean Absolute Error (MAE)**:
   - The average absolute difference between predicted and actual values.
   - **Why Use It?**:
     - Easy to interpret since it directly measures the average error in the same units as the target variable.
   - **Formula**:  
     $$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$  
   - **Minimization Goal**: Models strive to minimize this value.

2. **Mean Squared Error (MSE)**:
   - The average of squared differences between predicted and actual values.
   - **Why Use It?**:
     - Penalizes larger errors more heavily, making it sensitive to outliers.
   - **Formula**:  
     $$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$  
   - **Minimization Goal**: Lower values indicate better model accuracy.

3. **Root Mean Squared Error (RMSE)**:
   - Square root of MSE; provides errors in the same units as the target variable.
   - **Why Use It?**:
     - More intuitive than MSE as it is expressed in the same units as the data.
   - **Formula**:  
     $$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$  

---

### **Key Insight**
All of the above metrics are considered **loss functions**. In machine learning, the goal is to **minimize these loss functions** to improve model accuracy and reliability.
