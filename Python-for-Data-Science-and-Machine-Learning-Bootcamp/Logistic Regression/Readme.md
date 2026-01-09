# **🔹 Logistic Regression: A Method for Classification**

Logistic Regression is a **supervised learning algorithm** used for **classification tasks**, especially **binary classification**. It models the probability that a given input belongs to a particular class using the **sigmoid function**.

---

<img widht='600' height='400' src="https://github.com/user-attachments/assets/1da48caa-7645-4f7c-a3ce-31956ef8fa0f">

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

---

## **🔸 Multiclass Classification with Logistic Regression**

### **1️⃣ One-vs-Rest (OvR) / One-vs-All (OvA)**
- Train $k$ binary classifiers for $k$ classes  
- Each classifier predicts whether the input belongs to its class or not  
- Final prediction is the class with the **highest probability**

<img width='600' height='400' src="https://github.com/user-attachments/assets/c92f7114-4786-41f4-8740-f75a7e55bd28" />


### **2️⃣ Multinomial Logistic Regression (Softmax Regression)**
- Directly handles **multiclass classification**  
- Uses **softmax function** to compute probabilities:

$$
P(y = k \mid x) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}
$$

- Ensures that $\sum_{k=1}^{K} P(y = k \mid x) = 1$

<img width='600' height='400' src="https://github.com/user-attachments/assets/fa32aa9d-a9f3-425e-9027-d71ab93479e6" />

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

You can explore more on [Kaggle’s Logistic Regression competitions page](https://www.kaggle.com/competitions?tagIds=13404-Logistic+Regression).

---
