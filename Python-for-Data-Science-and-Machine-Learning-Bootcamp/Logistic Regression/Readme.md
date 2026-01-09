# **🔹 Logistic Regression: A Method for Classification**

Logistic Regression is a **supervised learning algorithm** used for **classification tasks**, especially **binary classification**. It models the probability that a given input belongs to a particular class using the **sigmoid function**.

---

<img width='600' height='400' src="https://github.com/user-attachments/assets/1da48caa-7645-4f7c-a3ce-31956ef8fa0f">

---

## **🔸 Examples of Binary Classification**
1. 📧 **Spam vs Ham Emails** → Classifies whether an email is spam or not  
2. 💰 **Loan Defaults** → Determines if a loan will default ($\text{Yes/No}$)  
3. 🏥 **Disease Diagnosis** → Predicts the presence or absence of a disease  

---

## **🔸 Sigmoid Function (Logistic Function)**
- The **sigmoid function** maps any real-valued number to a value between **0 and 1**, representing probability.
- **Equation**:

$$
\theta(z) = \frac{1}{1 + e^{-z}}
$$

- As $z \to \infty$, $\theta(z) \to 1$  
- As $z \to -\infty$, $\theta(z) \to 0$  

This makes logistic regression ideal for modeling probabilities.

---

## **🔸 Decision Boundary**
- Logistic regression outputs probabilities.  
- A **threshold** (commonly 0.5) is applied to decide the class label.  
- Example: If $P(y=1|x) > 0.5$, classify as **positive class**; otherwise, **negative class**.

---

## **🔸 Evaluation Metrics**

### 📊 **Confusion Matrix**
A table used to evaluate classification performance:
- $TP$ → True Positives (correctly predicted positive class)  
- $TN$ → True Negatives (correctly predicted negative class)  
- $FP$ → False Positives (Type I Error)  
- $FN$ → False Negatives (Type II Error)  

### ❌ **Error Rate**
- Measures the proportion of incorrect predictions:

$$
\text{Error Rate} = \frac{FP + FN}{\text{Total Predictions}}
$$

### ✅ Other Metrics
- **Accuracy**: $(TP + TN) / \text{Total Predictions}$  
- **Precision**: $TP / (TP + FP)$  
- **Recall (Sensitivity)**: $TP / (TP + FN)$  
- **F1 Score**: Harmonic mean of precision and recall.  

---

## **🔸 Types of Logistic Regression**

### 1️⃣ **Binary Logistic Regression**
- Two possible outcomes (Yes/No, 0/1).  
- Uses the sigmoid function.  

<img width="600" height="400" alt="LogisticRegression_43_1" src="https://github.com/user-attachments/assets/c47f02fe-d65a-4a61-b84e-4f3069926cd6" />


### 2️⃣ **Multiclass Logistic Regression**
- More than two categories.  
- Two main strategies:
  - **One-vs-Rest (OvR)** → Train $k$ binary classifiers for $k$ classes.  
  - **Multinomial Logistic Regression (Softmax Regression)** → Train a single model using the softmax function.  

<img width='600' height='400' src="https://github.com/user-attachments/assets/c92f7114-4786-41f4-8740-f75a7e55bd28" />

---

### 3️⃣ **Ordinal Logistic Regression**
- Target variable has **ordered categories** (e.g., satisfaction levels: poor, fair, good, excellent).  
- Models cumulative probabilities.  

<img width="600" height="400" alt="images" src="https://github.com/user-attachments/assets/99ab3bf4-6bdd-4f93-b874-8627c6244222" />


### 4️⃣ **Multinomial Logistic Regression (Nominal)**
- Target variable has **unordered categories** (e.g., predicting fruit type: apple, banana, orange).  
- Uses softmax to assign probabilities across classes.  

<img width='600' height='400' src="https://github.com/user-attachments/assets/fa32aa9d-a9f3-425e-9027-d71ab93479e6" />

---

## **🔸 Advantages of Logistic Regression**
- Simple and interpretable.  
- Outputs probabilities, not just class labels.  
- Efficient to train, even on large datasets.  
- Works well for linearly separable data.  
- Provides insights into feature importance via coefficients.  

---

## **🔸 Limitations**
- Assumes linear relationship between features and log-odds.  
- Struggles with complex, non-linear boundaries.  
- Sensitive to multicollinearity (correlated features).  
- Not ideal for very high-dimensional sparse data compared to SVMs or deep learning.  

---

## **🔸 Python Implementation Example**
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = LogisticRegression(multi_class='ovr', max_iter=1000)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
```

---

## **🔸 Kaggle Competitions Using Logistic Regression**
Logistic Regression is frequently used in Kaggle competitions for classification tasks. Some popular examples include:

| 🏆 Competition | 🔍 Task |
|----------------|--------|
| [Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions?tagIds=13404-Logistic+Regression) | Predict survival on the Titanic |
| [Sentiment Analysis on Movie Reviews](https://github.com/chapagain/kaggle-competitions-solution) | Classify sentiment from text |
| [Digit Recognizer](https://github.com/chapagain/kaggle-competitions-solution) | Identify handwritten digits |
| [Loan Default Prediction](https://github.com/jayinai/kaggle-regression) | Predict loan repayment behavior |
| [Restaurant Revenue Prediction](https://github.com/jayinai/kaggle-regression) | Forecast restaurant sales using features |

---

# ✅ Summary
- Logistic Regression is a **probabilistic classifier** using the **sigmoid function**.  
- Works best for **binary classification**, extended to multiclass via **OvR** or **Softmax**.  
- Variants include **binary, multinomial, and ordinal logistic regression**.  
- Evaluation uses confusion matrix, precision, recall, F1 score, and accuracy.  
- Advantages: simple, interpretable, efficient.  
- Limitations: assumes linearity, struggles with complex boundaries.  
- Still a strong baseline model in both academia and Kaggle competitions.  

---
