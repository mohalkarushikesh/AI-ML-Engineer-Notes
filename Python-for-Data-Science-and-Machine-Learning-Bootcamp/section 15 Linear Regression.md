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
