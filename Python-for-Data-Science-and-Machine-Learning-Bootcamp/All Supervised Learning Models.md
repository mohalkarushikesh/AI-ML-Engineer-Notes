Here’s your **ultra-short Machine Learning (ML) algorithms cheatsheet** 🚀:

### **Supervised Learning**
1️⃣ **Linear Regression** → Predicts continuous values.  
   _Ex:_ House price prediction.  


2️⃣ **Logistic Regression** → Predicts probabilities for classification.  
   _Ex:_ Spam detection.  


3️⃣ **Decision Tree** → Splits data into rules for classification.  
   _Ex:_ Loan approval system.  


4️⃣ **Random Forest** → Multiple decision trees for better predictions.  
   _Ex:_ Fraud detection.  


5️⃣ **Support Vector Machine (SVM)** → Finds optimal boundary for separation.  
   _Ex:_ Cancer cell classification.  


6️⃣ **Naïve Bayes** → Probabilistic classification using Bayes’ theorem.  
   _Ex:_ Sentiment analysis.  


7️⃣ **K-Nearest Neighbor (KNN)** → Classifies based on closest neighbors.  
   _Ex:_ Handwritten digit recognition.  


### **Unsupervised Learning**
8️⃣ **K-Means Clustering** → Groups similar data points into clusters.  
   _Ex:_ Customer segmentation.  


9️⃣ **Principal Component Analysis (PCA)** → Reduces dataset size while keeping variance.  
   _Ex:_ Face recognition.  


### **Deep Learning**
🔟 **Neural Networks** → Layers of artificial neurons for complex patterns.  
   _Ex:_ Image recognition & NLP.  

---

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
Here’s the equation for **K-Nearest Neighbor (KNN)**:
   1. The distance between two points \( A(X_1, Y_1) \) and \( B(X_2, Y_2) \) is often computed using the **Euclidean Distance** formula:
   
   $$d = \sqrt{(X_2 - X_1)^2 + (Y_2 - Y_1)^2}\$$

   Other distance metrics include:
   2. **Manhattan Distance**:
   
   $$d = |X_2 - X_1| + |Y_2 - Y_1|$$
   
   3. **Hamming Distance** (for categorical data):
   
   $$d = \sum (X_i \neq Y_i)$$

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

## **1️⃣ Decision Tree**
**📌 What It Is:**  
A hierarchical model that splits data into branches using decision rules based on feature values.

**📌 Equation:**  
The impurity measure for splitting nodes (Gini Index or Entropy) is calculated as:

- **Gini Index** (for classification):

$$Gini = 1 - \sum_{i=1}^{C} p_i^2$$

- **Entropy** (alternative impurity measure):

$$Entropy = - \sum_{i=1}^{C} p_i \log_2 p_i$$

Where \( p_i \) is the probability of class \( i \).

**📌 Use Case:**  
Used in **classification and regression tasks** where rule-based decisions work well.  
Example: **Fraud detection in banking**.

**📌 Types:**  
1. **Classification Trees** → Predicts discrete labels (e.g., Yes/No).  
2. **Regression Trees** → Predicts continuous values.  

**📌 Special Property:**  
Uses **pruning techniques** to prevent overfitting.

**📌 Evaluation Metrics:**  
- **Accuracy, Precision, Recall** (classification).  
- **Mean Squared Error (MSE)** (regression).  

**📌 Example:**  
Predicting **loan default risks** based on income, credit score, and loan amount.

---

## **2️⃣ Random Forest**
**📌 What It Is:**  
An ensemble learning model that builds multiple decision trees and averages their predictions.

**📌 Equation:**  
For classification, final prediction is based on majority voting:

$$Prediction = \text{Mode}(\hat{y}_1, \hat{y}_2, ..., \hat{y}_n)$$

For regression, it is based on the mean value:

$$Prediction = \frac{1}{N} \sum_{i=1}^{N} \hat{y}_i$$

Where \( \hat{y}_i \) represents predictions from individual trees.

**📌 Use Case:**  
Used for **high-dimensional datasets** where individual decision trees may overfit.  
Example: **Predicting customer churn in telecom**.

**📌 Special Property:**  
Uses **Bootstrap Aggregating (Bagging)** to improve stability.

**📌 Evaluation Metrics:**  
- **Accuracy, ROC-AUC** for classification.  
- **MSE, RMSE** for regression.  

**📌 Example:**  
Predicting whether an **email is spam or not**, using multiple word features.

---

## **3️⃣ Support Vector Machine (SVM)**
**📌 What It Is:**  
A model that finds the **optimal hyperplane** for separating classes.

**📌 Equation:**  
A hyperplane is defined as:

$$wX + b = 0$$

Where:
- \( w \) = Weight vector  
- \( X \) = Feature vector  
- \( b \) = Bias term

For classification, the goal is to **maximize the margin**:

$$\min \frac{1}{2} ||w||^2 \quad \text{subject to} \quad y_i(wX_i + b) \geq 1$$

**📌 Use Case:**  
Used in **text classification, bioinformatics, and image recognition**.  
Example: **Classifying cancer cells as benign or malignant**.

**📌 Types:**  
1. **Linear SVM** → Best for **linearly separable data**.  
2. **Non-Linear SVM** → Uses **Kernel Trick** for complex patterns.  

**📌 Special Property:**  
Maximizes **margin** to separate classes effectively.

**📌 Evaluation Metrics:**  
- **Precision, Recall, F1-score**.  
- **Hinge Loss** measures boundary correctness.

**📌 Example:**  
Classifying **customer reviews** as positive or negative.

---

## **4️⃣ Naïve Bayes**
**📌 What It Is:**  
A probabilistic model based on **Bayes’ Theorem**.

**📌 Equation:**  
Bayes’ Theorem is given by:

$$P(A|B) = \frac{P(B|A) P(A)}{P(B)}$$

For classification:

$$P(Y|X) = \frac{P(X|Y) P(Y)}{P(X)}$$

Where:
- \( P(A|B) \) = Posterior probability  
- \( P(B|A) \) = Likelihood  
- \( P(A) \) = Prior probability  
- \( P(B) \) = Evidence

**📌 Use Case:**  
Works well for **text classification, sentiment analysis, and spam filtering**.  
Example: **Detecting fake news articles based on text frequency**.

**📌 Types:**  
1. **Gaussian Naïve Bayes** → Assumes normal distribution for continuous data.  
2. **Multinomial Naïve Bayes** → Best for word frequency data.  
3. **Bernoulli Naïve Bayes** → Used for binary feature sets.

**📌 Special Property:**  
Assumes **independence between features** for fast computation.

**📌 Evaluation Metrics:**  
- **Log-Loss (Negative Log Likelihood)**.  
- **Accuracy, Precision, Recall**.

**📌 Example:**  
Classifying **news articles as real or fake** based on word occurrences.

---

## **5️⃣ K-Means Clustering**
**📌 What It Is:**  
An unsupervised algorithm that groups similar data points into **K clusters**.

**📌 Equation:**  
The cluster centroids are updated iteratively:

$$C_k = \frac{1}{N} \sum_{i=1}^{N} X_i$$

Where:
- \( C_k \) = Centroid of cluster \( k \)  
- \( X_i \) = Data point assigned to cluster  
- \( N \) = Number of data points in the cluster

**📌 Use Case:**  
Used in **customer segmentation, anomaly detection, and recommendation systems**.  
Example: **Grouping customers based on spending habits**.

**📌 Special Property:**  
Uses **Euclidean Distance** for cluster assignment:

$$d = \sqrt{(X_2 - X_1)^2 + (Y_2 - Y_1)^2}$$

**📌 Evaluation Metrics:**  
- **Silhouette Score** → Measures cluster cohesion.  
- **WCSS (Within Cluster Sum of Squares)** → Measures intra-cluster variance.

**📌 Example:**  
Segmenting **customers into high-spenders, mid-range, and budget shoppers**.

---

## **6️⃣ Principal Component Analysis (PCA)**
**📌 What It Is:**  
A dimensionality reduction technique that removes redundancy while preserving variance.

**📌 Equation:**  
Eigenvalue decomposition of covariance matrix:

$$X' = W X$$

Where:
- \( W \) = Transformation matrix  
- \( X \) = Original dataset  
- \( X' \) = Reduced feature representation

**📌 Use Case:**  
Used for **reducing dataset size** while retaining variance.  
Example: **Facial recognition in security systems**.

**📌 Special Property:**  
- Converts correlated features into a set of **independent principal components**.  
- Retains **maximum variance** in fewer dimensions.

**📌 Evaluation Metrics:**  
- **Explained Variance Ratio** → Indicates information retained.  
- **Reconstruction Error** → Measures loss after compression.

**📌 Example:**  
Compressing **high-resolution images** into fewer dimensions.

---

## **7️⃣ Neural Networks (Deep Learning)**
**📌 What It Is:**  
A powerful model that mimics the human brain using layers of interconnected neurons.

**📌 Equation:**  
A neural network's forward propagation is calculated as:

$$Z = W X + b$$

Activation function:

$$A = \sigma(Z)$$

Where:
- \( W \) = Weights  
- \( X \) = Input data  
- \( b \) = Bias  
- \( \sigma \) = Activation function (ReLU, Sigmoid, Softmax)

**📌 Use Case:**  
Used in **image recognition, speech processing, and language translation**.  
Example: **Predicting handwritten digits in OCR systems**.

**📌 Special Property:**  
Uses **backpropagation and gradient descent** for learning.

**📌 Evaluation Metrics:**  
- **Cross-Entropy Loss** (classification).  
- **MSE** (regression).  

**📌 Example:**  
Generating **realistic human faces** in AI-powered image synthesis.

---
