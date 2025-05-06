### **Linear Regression**
- **Predicts continuous values** by fitting a straight line to the data.
- It assumes a **linear relationship** between independent (\(X\)) and dependent (\(Y\)) variables.

#### **📌 Equation**

$$Y = \beta_0 + \beta_1X + \epsilon$$

Where:
- \(Y\) → Dependent variable (target)
- \(X\) → Independent variable (input)
- \(\beta_0, \beta_1\) → Model parameters (intercept & slope)
- \(\epsilon\) → Error term (random noise)

---

### **🔹 Use Case**
Used for predicting **continuous numerical values**.
- **Example**: Predicting **house prices** based on **area** and **number of rooms**.

---

### **🔹 Types of Linear Regression**
1. **Simple Linear Regression** → Uses **one independent variable**.
   - Example: Predicting **salary** based on **years of experience**.
2. **Multiple Linear Regression** → Uses **two or more independent variables**.
   - Example: Predicting **house prices** based on **size, location, and number of bedrooms**.
3. **Polynomial Regression** → Extends linear regression by adding polynomial terms to capture **non-linearity**.
   - Example: Modeling **growth rate** of bacteria over time.
4. **Ridge Regression (L2 Regularization)** → Helps handle **multicollinearity** by shrinking large coefficients.
   - Example: Predicting **stock prices** with multiple correlated financial factors.
5. **Lasso Regression (L1 Regularization)** → Shrinks some coefficients to **zero**, helping in **feature selection**.
   - Example: Selecting **key factors** affecting **customer satisfaction**.
6. **Elastic Net Regression** → Combination of **Ridge & Lasso**, balancing feature selection & coefficient shrinkage.
   - Example: **Predicting sales revenue** with multiple interrelated factors.
7. **Stepwise Regression** → Adds/removes predictors systematically based on significance.
   - Example: **Medical diagnosis models** where only important patient data is selected.
8. **Quantile Regression** → Estimates percentiles instead of mean values.
   - Example: **House price estimation for different market segments**.
9. **Bayesian Regression** → Uses probability distributions to estimate coefficients.
   - Example: **Weather forecasting with uncertain data**.
---

### **🔹 Evaluation Metrics**
1. **Mean Absolute Error (MAE)** → Measures average absolute difference between actual and predicted values.
2. **Mean Squared Error (MSE)** → Penalizes larger errors more heavily.
3. **Root Mean Squared Error (RMSE)** → The square root of MSE; expressed in the same units as the target variable.

---

### **🔹 Example**
**Predicting house prices**  
- **Independent Variables (X)** → Square footage, number of rooms  
- **Dependent Variable (Y)** → Price  
- The best-fit regression line estimates house prices based on **historical data trends**.

---

### **Logistic Regression**
- **Predicts probabilities** for categorical outcomes (classification task).
- Unlike Linear Regression, it models data using the **logistic (sigmoid) function** to restrict predictions between 0 and 1.

#### **📌 Equation**

$$p(Y=1) = \frac{1}{1+e^{-(\beta_0 + \beta_1X)}}$$

Where:
- \(p(Y=1)\) → Probability that the outcome belongs to class 1
- \(\beta_0, \beta_1\) → Model parameters
- \(X\) → Independent variable
- \(e\) → Euler’s number (≈ 2.718)

---

### **🔹 Use Case**
Used for classification problems, typically **binary (Yes/No, True/False) classification**.
- **Example**: Predicting whether a **customer will purchase a product** based on browsing behavior.

---

### **🔹 Types of Logistic Regression**
1. **Binary Logistic Regression** → Two possible outcomes (e.g., Spam vs. Not Spam).
2. **Multinomial Logistic Regression** → Three or more unordered categories (e.g., choosing between multiple brands).
3. **Ordinal Logistic Regression** → Three or more ordered categories (e.g., ranking customer satisfaction as Low, Medium, High).

---

### **🔹 Special Property**
- Uses the **sigmoid function** to convert raw predictions into probability values.
- If \(p(Y=1) > 0.5\), classify as **Class 1**; else, classify as **Class 0**.

---

### **🔹 Evaluation Metrics**
1. **Accuracy** → Percentage of correct predictions.
2. **Precision** → Correct **positive predictions** over all predicted positives.
3. **Recall** → Correct **positive predictions** over all actual positives.
4. **F1-score** → Harmonic mean of **precision and recall**.
5. **ROC Curve & AUC** → Measures model’s performance in differentiating classes.

---

### **🔹 Example**
**Predicting if a student will pass an exam**  
- **X (Independent Variable)** → Hours studied  
- **Y (Dependent Variable)** → Pass/Fail (1 or 0)  
- If the model predicts \(p(Y=1) = 0.85\), student is classified as **"Will Pass"**.

---

### **K-Nearest Neighbor (KNN)**
- **Works for both classification & regression tasks**.
- A **lazy algorithm** (doesn’t build a model; instead, makes predictions using stored data points).

#### **📌 How It Works**
1. **Calculate distances** between test data and stored data points (Euclidean, Manhattan, Hamming).
2. **Select K nearest points**:
   - **K ≈ sqrt(n)** (where \(n\) is the number of data points).
   - **Odd K** is preferred to avoid ties between two classes.
3. **Determine category (classification) or mean value (regression).**

---

### **🔹 Use Case**
Used in tasks where instances must be classified based on **similarity to known examples**.
- **Example**: Identifying **handwritten digits** by comparing with stored examples.

---

### **🔹 Special Property**
- Doesn’t assume **underlying data distribution**.
- Choosing **K** affects accuracy:  
  - **Small K** → More variance, sensitive to noise.  
  - **Large K** → More bias, less flexibility.

---

### **🔹 Types of K-Nearest Neighbor **
1. **KNN for Classification** → Assigns labels based on the majority of nearest neighbors.
   - Example: **Handwritten digit recognition**.
2. **KNN for Regression** → Estimates a continuous value by averaging nearest neighbors.
   - Example: **Predicting temperature based on nearby weather stations**.
3. **Weighted KNN** → Gives more influence to closer neighbors using weighted distance.
   - Example: **Personalized recommendation systems**.
4. **KNN for Clustering (unsupervised learning)** → Groups similar data points together.
   - Example: **Customer segmentation in marketing**.

---
### **🔹 Evaluation Metrics**
1. **Accuracy** → Correct classifications.
2. **Confusion Matrix** → Breakdown of correct and incorrect predictions.
3. **RMSE (if used for regression tasks)** → Measures prediction error.
4. **Silhouette Score (for clustering)** → Evaluates how well data points are grouped.

---

### **🔹 Example**
**Classifying fruit types based on size and color**
- **Features (X):** Diameter, color intensity.
- **Labels (Y):** Apple, Orange, Banana.
- A new fruit is classified **based on the closest K samples** in the dataset.

---
