### ML Process 
1. Data Acquisition

2. Data cleaning (EDA)
   - Read data
   - Correlation (jointplot)
   - Relationships (pairplot)
   - Linear model plot (seaborn lmplot)  
     Example: `sns.lmplot(data=customers, x='', y='')`

3. Split the data (Training/Testing)

   `y = (what you want to predict)`  
   `x = (factors that influence y)`

   Where y will be predicted and x is numerical features.

   Splitting the data:
   ```python
   from sklearn.cross_validation import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
   ```

4. Train the model (Linear Regression)
   ```python
   from sklearn.linear_model import LinearRegression

   lm = LinearRegression()  # Create the instance

   lm.fit(X_train, y_train)  # Train/fit lm on training data

   lm.coef_  # Print coefficients
   ```

5. Model testing (Feed the test data)
   ```python
   predictions = lm.predict(X_test)  # Predictions

   plt.scatterplot(y_test, predictions)  # Plot (actual vs predicted values)
   plt.xlabel('Actual')
   plt.ylabel('Predicted')
   ```

   Evaluate:
   ```python
   from sklearn import metrics
   print('MAE:', metrics.mean_absolute_error(y_test, predictions))
   print('MSE:', metrics.mean_squared_error(y_test, predictions))
   print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

   metrics.explained_variance_score(y_test, predictions)
   ```

   Residuals:
   ```python
   sns.distplot((y_test - predictions), bins=50)
   # or use plt.hist()
   ```

6. Model Deployment

---

### ML (https://medium.com/@RobuRishabh/introduction-to-machine-learning-555b0f1b62f5)

#### Supervised learning:
Model learns from labeled data, meaning each input (data) has a corresponding output (label). The model learns the relationship between the inputs and outputs and can predict the outputs for new, unseen data.

- **Example**: House price prediction  
- **Analogy**: Student learning math with a teacher  
- **Goal**: Learn a function to map inputs to outputs  

---

```python
# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Create a simple dataset for house prices
# Features: size (in square feet), number of rooms
# Target: house price in $1000s
data = {
    'size': [1500, 1800, 2400, 3000, 3500, 4000, 4500, 5000, 5500, 6000],
    'rooms': [3, 3, 4, 4, 5, 5, 6, 6, 6, 7],
    'price': [300, 320, 400, 450, 500, 540, 600, 620, 670, 700]
}

# Convert the dictionary to a pandas DataFrame
df = pd.DataFrame(data)

# Define the features (X) and the target (y)
X = df[['size', 'rooms']]  # Input features
y = df['price']  # Target variable

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

# Output the results
print(f"Predicted prices: {y_pred}")
print(f"Actual prices: {y_test.values}")
print(f"Mean Squared Error: {mse:.2f}")
```

#### Unsupervised learning:
The data does not have labels. The model is only given the inputs and must find patterns and relationships between them.

- **Example**: Images of different animals  
- **Analogy**: Person classifying fruits in a basket based on color without knowing their names  
- **Goal**: Find patterns and groupings in data  

#### Reinforcement learning:
Involves an agent (model) which learns from trial and error while interacting with an environment. The agent receives rewards for good actions and penalties for bad actions.

- **Example**: A video game  
- **Analogy**: Teaching a pet a new trick  
- **Goal**: Learn through rewards and penalties  

---

### Cost Function
Cost function (aka loss function or error function):

1. For **Linear Regression**:  
   **MSE (Mean Squared Error)**: Measures the differences between the actual and predicted values, squares them (to avoid negative differences), and averages them over the dataset.

   ![MSE](https://miro.medium.com/v2/resize:fit:828/format:webp/1*B1UvTTDIfIa5hjupQS61YA.png)

3. For **Classification**:  
   **Logarithmic loss (Cross Entropy Loss)**: Measures the error in classifying between categories.

   ![LRL/CEL](https://miro.medium.com/v2/resize:fit:828/format:webp/1*aVBNdQsTaJwz7iyaeOtN4g.png)
   
---

### Gradient Descent
Gradient Descent is an analytical method used to find the optimal values for parameters (i.e., coefficients).

---

#### MSE:
Mean Squared Error calculates how well the model performs. The lower the MSE, the better the model predictions.

---

### Two Types of Tasks in ML
1. **Classification**: Output is discrete.
2. **Regression**: Output is continuous.

---

#### Linear Regression:
Represents the relationship between independent variable (input) and dependent variable (output).  
![LR](https://miro.medium.com/v2/resize:fit:828/format:webp/1*fXEiGgyFUdS1IVF51V5g6A.png)

---

#### Multivariable Linear Regression:
Uses multiple features (variables) to predict the output.  
![MLR](https://miro.medium.com/v2/resize:fit:828/format:webp/1*MtqZ3ukoUwlG2NgvliOkBg.png)
```
### Limitations of Linear Regression
1. **Assumption of Linearity**: Linear regression assumes a straight-line relationship between dependent and independent variables. However, real-world data is rarely linearly separable, and this assumption often leads to inaccuracies.
2. **Prone to Noise and Overfitting**: When the number of observations is less than the number of features, linear regression can result in overfitting, making the model excessively noisy.
3. **Prone to Outliers**: Outliers—data points that don't have clear relationships with other points—can significantly impact the accuracy of the model.

---

### Gradient Descent
Gradient Descent is an optimization algorithm used to minimize the cost function by iteratively updating parameters (coefficients). It works by gradually moving towards the optimal solution by following the slope of the cost function.

**Analogy**: Imagine you’re on a mountain and want to reach the lowest point (valley). In this case:
- The mountain represents the cost function.
- The goal is to minimize the cost and reach the valley.
- Each step represents minimizing the parameters (coefficients).

#### How Gradient Descent Works
1. **Initialize Parameters**: Start by assigning random values to parameters (coefficients).
2. **Compute the Cost Function**: Calculate the error between the predicted values and actual values, often using Mean Squared Error (MSE).
3. **Calculate Gradient (Slope)**: Compute the slope of the cost function with respect to each parameter. The slope indicates the direction to adjust parameters to reduce the cost.
4. **Update Parameters**: Adjust the parameters using a learning rate, which controls the size of the steps.
5. **Repeat**: Continue the process until the parameters converge, meaning further updates make minimal improvements.

---

### Importance of Learning Rate
Learning rate is a crucial hyperparameter that determines the step size in each iteration, controlling how much the coefficients change.

- **Small Learning Rate**: Leads to slow convergence, but ensures precision.
- **Large Learning Rate**: Allows faster convergence but risks overshooting the optimal solution, preventing the model from finding the minimum.

![LR](https://miro.medium.com/v2/resize:fit:604/format:webp/1*IgmACHtWHHqGTISYMWQyXw.png)

Feature scaling in machine learning



