# Logistic Regression: A Method for Classification

## Examples of Binary Classification
1. **Spam vs Ham Emails**: Classifies whether an email is spam or not.
2. **Loan Defaults**: Determines if a loan will default (Yes/No).
3. **Disease Diagnosis**: Predicts the presence or absence of a disease.

## Sigmoid Function (Logistic Function)
- The sigmoid function takes any input value and maps it to a range between **0 and 1**.
- **Equation**:  
  $$\theta(z) = \frac{1}{1 + e^{-z}}$$  

## Evaluation Metrics
### Confusion Matrix
A table used to evaluate the performance of a classification algorithm, containing:
- **True Positives (TP)**: Correctly classified positive instances.
- **True Negatives (TN)**: Correctly classified negative instances.
- **False Positives (FP)** (Type 1 Error): Incorrectly classified negative instances as positive.
- **False Negatives (FN)** (Type 2 Error): Incorrectly classified positive instances as negative.

### Error Rates
- **Misclassification Rate (Error Rate)**: The proportion of incorrect predictions out of the total predictions.  
  $$\text{Error Rate} = \frac{FP + FN}{Total Predictions}$$

- https://medium.com/@abhishekjainindore24/all-about-logistic-regression-bd135b6e3993

**Logistic Regression for multiclass classification**
- **One-vs-Rest (OvR) or One-vs-All (OvA)**: In this strategy, you train multiple binary classifiers, each representing one class against the rest. For k classes, you would train k separate binary Logistic Regression classifiers. During prediction, each classifier outputs a probability, and the class with the highest probability is chosen as the predicted class.

**Multimodel Logistic Regression(Softmax Regression)**
- Unlike the One-vs-Rest approach, Multinomial Logistic Regression directly extends the binary Logistic Regression to handle multiple classes.
- It uses the softmax activation function to calculate probabilities for each class. The softmax function converts raw scores (logits) into probabilities, ensuring that the sum of probabilities across all classes equals 1.
- In scikit-learn, the LogisticRegression class supports both binary and multiclass classification
```
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the Iris dataset (a classic multiclass classification dataset)
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Logistic Regression model
model = LogisticRegression(multi_class='ovr', max_iter=1000)  # 'ovr' is the default strategy for multiclass
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")  
```

Kaggle competions 
