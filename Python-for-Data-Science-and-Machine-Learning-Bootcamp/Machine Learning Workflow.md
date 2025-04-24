# Machine Learning Workflow

## 1. Data Acquisition
- Collect the required data for your analysis and model building.

## 2. Data Cleaning (EDA - Exploratory Data Analysis)
- **Read Data**: Load your dataset into the environment.
- **Correlation Analysis**: Use a `jointplot` to examine correlations.
- **Explore Relationships**: Use a `pairplot` to study feature relationships.
- **Linear Model Plot**: Utilize `sns.lmplot()` for visualizing linear relationships.

  Example:
  ```python
  sns.lmplot(data=customers, x='', y='')
  ```

## 3. Split the Data (Training / Testing)
- **Define Variables**:
  - `y`: The target variable (what you want to predict).
  - `X`: Features (factors influencing `y`).

- **Splitting the Data**:
  ```python
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
  ```

## 4. Train the Model (Linear Regression)
- **Create an Instance**:
  ```python
  from sklearn.linear_model import LinearRegression
  lm = LinearRegression()
  ```

- **Fit the Model**:
  ```python
  lm.fit(X_train, y_train)
  ```

- **Retrieve Coefficients**:
  ```python
  lm.coef_
  ```

## 5. Model Testing (Feeding Test Data)
- **Predictions**:
  ```python
  predictions = lm.predict(X_test)
  ```

- **Plot Actual vs Predicted Values**:
  ```python
  plt.scatter(y_test, predictions)
  plt.xlabel('Actual')
  plt.ylabel('Predicted')
  ```

### Evaluation
- **Calculate Metrics**:
  ```python
  from sklearn import metrics
  print('MAE:', metrics.mean_absolute_error(y_test, predictions))
  print('MSE:', metrics.mean_squared_error(y_test, predictions))
  print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
  ```

- **Variance Score**:
  ```python
  metrics.explained_variance_score(y_test, predictions)
  ```

### Residuals
- **Distribution of Errors**:
  ```python
  sns.distplot((y_test - predictions), bins=50)
  # OR
  plt.hist(y_test - predictions, bins=50)
  ```

## 6. Model Deployment
- Document and deploy the trained model for real-world predictions.

---

