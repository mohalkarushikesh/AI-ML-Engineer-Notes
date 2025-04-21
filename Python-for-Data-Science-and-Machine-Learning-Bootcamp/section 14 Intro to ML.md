## **Supervised Learning**

Supervised learning algorithms are trained using labeled examples where the desired output is known. 

### **Examples**
- A segment of text could have a category label such as:
  - **Spam vs. Legitimate Email**
  - **Positive vs. Negative Movie Review**

### **How It Works**
1. The network receives a set of inputs along with corresponding correct outputs.
2. The algorithm learns by comparing its actual output with the correct outputs to identify errors.
3. The model adjusts itself to reduce errors by modifying its parameters accordingly.

### **Applications**
Supervised learning is widely used in scenarios where historical data helps predict future outcomes. Examples include:
- Email filtering
- Sentiment analysis
- Stock price prediction
- Fraud detection

---

## **Machine Learning Process**

1. **Data Acquisition**: Collect relevant and high-quality data.
2. **Data Cleaning**: Process the data to remove inconsistencies, errors, or missing values.
3. **Model Training and Building**: Use the training data to teach the model patterns and relationships.
4. **Model Testing**: Validate the model's performance on unseen test data.
5. **Model Deployment**: Deploy the final model to real-world environments for practical use.

![ML-ProcessS](https://cdn.elearningindustry.com/wp-content/uploads/2017/05/73348f2f23b70566eef2d9f10f9fe22c-768x438.png)

---

## **Data Sets**

In machine learning, three key datasets are used:

- **Training Data**: Used to train the model by allowing it to learn patterns and relationships.
- **Validation Data**: Helps fine-tune model parameters and avoid overfitting.
- **Test Data**: Provides a final unbiased evaluation of the model's performance.

---

## **Evaluating Performance**

Performance metrics are crucial to assess how well a machine learning model performs.

### **Key Metrics for Classification**

1. **Accuracy**:
   - Formula:  
     $$Accuracy = \frac{{\text{Number of Correct Predictions}}}{{\text{Total Number of Predictions}}}$$
   - Useful for well-balanced datasets.
   - Not suitable for imbalanced datasets (e.g., detecting rare diseases).

2. **Recall (Sensitivity)**:
   - Measures the model's ability to find all relevant positive cases.
   - Formula:  
     $$Recall = \frac{{\text{True Positives}}}{{\text{True Positives + False Negatives}}}$$

3. **Precision**:
   - Indicates how many of the predicted positive results are actually relevant.
   - Formula:  
     $$Precision = \frac{{\text{True Positives}}}{{\text{True Positives + False Positives}}}$$

4. **F1-Score**:
   - The harmonic mean of precision and recall.
   - Formula:  
     $$F1 = 2 * \frac{{\text{Precision} \cdot \text{Recall}}}{{\text{Precision} + \text{Recall}}}$$
   - Ideal for imbalanced datasets where both precision and recall are important.

---

## **Binary Classification: Confusion Matrix**

The confusion matrix evaluates a classification model by visualizing prediction results. It consists of:

| **Actual\Predicted** | **Positive** | **Negative** |
|-----------------------|--------------|--------------|
| **Positive**          | True Positive (TP) | False Negative (FN) |
| **Negative**          | False Positive (FP) | True Negative (TN) |

---

## **Evaluating Performance: Regression**

For regression tasks, we use different performance metrics:

1. **Mean Absolute Error (MAE)**:
   - Measures the average magnitude of errors between predicted and actual values.

2. **Mean Squared Error (MSE)**:
   - Penalizes larger errors more than MAE by squaring the differences.

3. **Root Mean Squared Error (RMSE)**:
   - Square root of MSE; gives errors in the same units as the target variable.

4. **R² Score (Coefficient of Determination)**:
   - Indicates the proportion of variance in the target variable explained by the model.
   - Formula:  
     $$R^2 = 1 - \frac{{\text{Sum of Squared Residuals (SSR)}}}{{\text{Total Sum of Squares (TSS)}}}$$

---
