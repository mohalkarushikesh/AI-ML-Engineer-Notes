---

**Supervised Learning**

Supervised learning algorithms are trained using labeled examples where the desired output is known. 

For example:
- A segment of text could have a category label such as:
  - Spam vs. Legitimate Email
  - Positive vs. Negative Movie Review

How it works:
- The network receives a set of inputs along with corresponding correct outputs.
- The algorithm learns by comparing its actual output with the correct outputs to identify errors.
- It then modifies the model accordingly.

Applications:
- Supervised learning is commonly used in scenarios where historical data is used to predict likely future events.

---

**Machine Learning Process**

1. **Data Acquisition**: Collecting relevant data.
2. **Data Cleaning**: Ensuring the data is free of errors and inconsistencies.
3. **Model Training and Building**: Creating the model by learning patterns in the training data.
4. **Model Testing**: Evaluating the model's performance using test data.
5. **Model Deployment**: Implementing the model in real-world applications.

![ML-ProcessS](https://cdn.elearningindustry.com/wp-content/uploads/2017/05/73348f2f23b70566eef2d9f10f9fe22c-768x438.png)

---

**Data Sets**

Machine learning involves three key sets of data:
- **Training Data**: Used to train the model.
- **Validation Data**: Helps adjust model parameters.
- **Test Data**: Used to evaluate the final performance of the model.

---

**Evaluating Performance (Classification)**

Performance metrics evaluate how well the model performs, especially in classification problems.

**Key Metrics**:
1. **Accuracy**:
   - The number of correct predictions divided by the total number of predictions.
   - Useful when target classes are well-balanced.
   - Not ideal for unbalanced classes.

2. **Recall**:
   - The ability of the model to find all relevant cases within the dataset.

3. **Precision**:
   - The ability of the classification model to identify only relevant data points.

4. **F1-Score**:
   - Combines precision and recall to provide a balanced metric.
   - Formula:  
     $$F1 = 2 * \frac{{\text{Precision} \cdot \text{Recall}}}{{\text{Precision} + \text{Recall}}}$$
   - Useful in scenarios requiring an optimal blend of precision and recall.

---

**Binary Classification: Confusion Matrix**

The confusion matrix is commonly used in binary classification tasks to evaluate model performance. It provides detailed insight into prediction results across different classes.
