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

Kaggle competions 
