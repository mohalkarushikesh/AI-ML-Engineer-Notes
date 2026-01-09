## **K-Nearest Neighbor (KNN)**

K-Nearest Neighbors (KNN) is a powerful and intuitive supervised learning algorithm used for both **classification** and **regression tasks**. The fundamental idea is to predict the correct class or value for test data points by finding **K nearest training points** based on distance.

### **Key Idea to choose a value of k **
KNN predicts the category or value by:
1. **Calculating distances** between the test data point and all training points.
2. **Selecting K closest points** (neighbors).
- sqrt(n), where n is the total number of data points 
- odd value of k is selected to avoid confusion between two classes of data
3. Determining the category or mean value among those neighbors.

For **classification**, KNN picks the category with the majority vote among K neighbors. For **regression**, it takes the mean value.

![KNN Working](https://miro.medium.com/v2/resize:fit:828/format:webp/0*34SajbTO2C5Lvigs.png)

#### **Example**:
Imagine a scenario where we have an image resembling both a cat and a dog. To determine whether it is a cat or dog, KNN compares features of the new image with datasets of cats and dogs and places it in the most similar category.

---

### **Why is K-NN Needed?**

Consider we have two categories, **Category A** and **Category B**, with a new data point `x1`. Which category does `x1` belong to? KNN can efficiently classify this data point based on its proximity to the training data points of each category.

![Illustration](https://miro.medium.com/v2/resize:fit:720/format:webp/0*OltO4Txr-D0lPWNL.png)

---

### **Outliers**

**Outliers** are data points that deviate significantly from expected patterns. These can impact KNN model performance and may lead to incorrect classifications or predictions. Identifying and handling outliers is critical to improving KNN results.

---

### **Algorithm Steps**

1. **Select the number of neighbors (K)**.
2. **Calculate distances** between the new point and all training points. Distance metrics include:
   - **Euclidean Distance**: Straight-line distance between two points.
   - **Manhattan Distance**: Sum of absolute differences between vectors.
   - **Hamming Distance**: Used for categorical variables; assigns 0 for matching values and 1 for non-matching values.

![Distance Metrics](https://miro.medium.com/v2/resize:fit:600/format:webp/0*cc0YOp9aRJzFpVHa.png)

![Hamming Distance](https://miro.medium.com/v2/resize:fit:432/format:webp/0*qxYGj_pO6Qj_coZ_.png)

3. **Select the K nearest neighbors** based on distance.
4. **Count the number of data points** in each category among neighbors.
5. **Assign the new point** to the category with the maximum neighbors.
6. **Model is ready to predict!**

---

### **Optimal K Value**

Selecting the best K value is important:
- **Small K values** lead to unstable, overly sensitive decision boundaries.
- **Large K values** smoothen decision boundaries but can reduce precision.

#### **How to Find Optimal K**:
- Experiment with different K values and compute error rates.
- Plot **Error Rate vs. K Value** and select K with minimum error.

---

### **Methods for Implementing KNN**

#### **KNeighborsClassifier Parameters**:
```python
KNeighborsClassifier(n_neighbors=5, *, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None, **kwargs)
```
- **Algorithm Options**: {'auto', 'ball_tree', 'kd_tree', 'brute'}
  - **Brute Force**: Considers all data points for classification; ideal for small datasets.
  - **KD Tree**: Organizes points hierarchically for faster processing; useful for large datasets.
  - **Ball Tree**: An efficient structure similar to KD Tree; handles larger datasets well.

#### **Comparison Summary**:
| Method        | Usage          | Advantages          | Drawbacks              |
|---------------|----------------|---------------------|------------------------|
| **Brute Force** | Small datasets | High accuracy       | Slow for large datasets |
| **KD Tree**    | Large datasets | Speed & efficiency  | Requires structured data |
| **Ball Tree**  | Large datasets | Speed & efficiency  | Requires structured data |

---

### **KNN Model Implementation**

#### **Code Example**
```python
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing, metrics

# Load data
df = pd.read_csv('Telecustomers.csv')
X = df.drop(['custcat'], axis=1)
y = df['custcat']

# Standardize data
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

# Train model
k = 4
neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
Pred_y = neigh.predict(X_test)

# Evaluate model accuracy
print("Accuracy at K=4:", metrics.accuracy_score(y_test, Pred_y))
```

#### **Error Rate vs. K Value**
```python
error_rate = []
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10, 6))
plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
print("Minimum error:", min(error_rate), "at K =", error_rate.index(min(error_rate)))
```

#### **Accuracy vs. K Value**
```python
acc = []
for i in range(1, 40):
    neigh = KNeighborsClassifier(n_neighbors=i).fit(X_train, y_train)
    yhat = neigh.predict(X_test)
    acc.append(metrics.accuracy_score(y_test, yhat))

plt.figure(figsize=(10, 6))
plt.plot(range(1, 40), acc, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.title('Accuracy vs. K Value')
plt.xlabel('K')
plt.ylabel('Accuracy')
print("Maximum accuracy:", max(acc), "at K =", acc.index(max(acc)))
```

---

### **Key Insights**
- KNN is versatile and effective for many classification and regression tasks.
- The choice of K and distance metrics impacts performance greatly.
- For large datasets, KD Tree or Ball Tree methods offer efficiency.

Let me know if you'd like more insights!
