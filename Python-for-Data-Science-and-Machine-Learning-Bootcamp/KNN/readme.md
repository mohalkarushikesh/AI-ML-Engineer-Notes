This script implements **K-Nearest Neighbors (KNN)** classification using **Scikit-learn** on a dataset named `"Classified Data"`. Here's a breakdown of each step:

### **1. Importing Libraries**
The script imports necessary libraries for data processing and visualization:
- `pandas` for handling tabular data.
- `numpy` for numerical operations.
- `matplotlib.pyplot` and `seaborn` for visualizing the results.
- `%matplotlib inline` ensures plots render in Jupyter Notebook.

### **2. Loading and Preparing Data**
- The dataset `"Classified Data"` is read using `pd.read_csv()`, setting the first column as an index.
- The `StandardScaler()` is used to **normalize** the features (excluding the target class) to standardize the dataset:
  ```python
  scalar.fit(df.drop('TARGET CLASS', axis=1))
  scaled_features = scalar.transform(df.drop('TARGET CLASS', axis=1))
  ```
- A new DataFrame `df_feat` is created with these transformed features, maintaining the original column names.

### **3. Splitting the Dataset**
The dataset is split into **training and testing sets**:
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
```
- `X`: Features (standardized).
- `y`: Target variable (`TARGET CLASS`).
- `train_test_split()`: Allocates 70% for training and 30% for testing.

### **4. Training KNN Model**
- A **KNN classifier** is created with `n_neighbors=1` (using only one neighbor for classification).
- The classifier is trained using:
  ```python
  knn.fit(X_train, y_train)
  ```
- Predictions are made on the test set.

### **5. Evaluating the Model**
- `classification_report(y_test, pred)` shows precision, recall, and F1-score.
- `confusion_matrix(y_test, pred)` prints a confusion matrix to analyze misclassifications.

### **6. Finding the Optimal `K` Value**
Instead of using `k=1`, a **loop evaluates K from 1 to 40**, measuring error rate:
```python
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
```
- **Error rate** is calculated as the proportion of misclassified instances.
- Results are plotted to visualize how `K` impacts classification accuracy.

### **7. Selecting Optimal `K`**
Based on the **error rate plot**, `K=17` is chosen:
```python
knn = KNeighborsClassifier(n_neighbors=17)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
```
- A final **classification report and confusion matrix** are generated to evaluate model performance.

### **Summary**
- The script implements **KNN classification** on standardized data.
- The **best K value is determined** through an error rate comparison.
- **KNN performance** is evaluated using precision, recall, and confusion matrices.

