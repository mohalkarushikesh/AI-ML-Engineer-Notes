[**K-Nearest Neighbor**


K-nearest neighbors (KNN) is a type of supervised learning algorithm used for both regression and classification.KNN tries to predict the correct class for the test data by calculating the distance between the test data and all the training points. Then select the K number of points which is closet to the test data. The KNN algorithm calculates the probability of the test data belonging to the classes of ‘K’ training data and class holds the highest probability will be selected. In the case of regression, the value is the mean of the ‘K’ selected training points.



Suppose, we have an image of a creature that looks similar to cat and dog, but we want to know either it is a cat or dog. So for this identification, we can use the KNN algorithm, as it works on a similarity measure. Our KNN model will find the similar features of the new data set to the cats and dogs images and based on the most similar features it will put it in either cat or dog category.

Why do we need a K-NN Algorithm?

Suppose there are two categories, i.e., Category A and Category B, and we have a new data point x1, so this data point will lie in which of these categories. To solve this type of problem, we need a K-NN algorithm. With the help of K-NN, we can easily identify the category or class of a particular dataset. Consider the below diagram:

![img](https://miro.medium.com/v2/resize:fit:720/format:webp/0*OltO4Txr-D0lPWNL.png)

Outliers: are data points that significantly deviate from expected pattern potentially impacting model performances and leading to incorrect inferences 

The K-NN working can be explained on the basis of the below algorithm:

Step-1: Select the number K of the neighbors
Step-2: Calculate the Euclidean distance of K number of neighbors

  - The first step is to calculate the distance between the new point and each training point
  - Euclidian, Manhattan (for continuous) and Hamming distance (for categorical).
  - Euclidean Distance: Euclidean distance is calculated as the square root of the sum of the squared differences between a new point (x) and an existing point (y).
  - Manhattan Distance: This is the distance between real vectors using the sum of their absolute difference.
  ![euclidian and manhattan](https://miro.medium.com/v2/resize:fit:600/format:webp/0*cc0YOp9aRJzFpVHa.png)
  - Hamming Distance: It is used for categorical variables. If the value (x) and the value (y) are the same, the distance D will be equal to 0 . Otherwise D=1.
  - ![](https://miro.medium.com/v2/resize:fit:432/format:webp/0*qxYGj_pO6Qj_coZ_.png)
Step-3: Take the K nearest neighbors as per the calculated Euclidean distance.
Step-4: Among these k neighbors, count the number of the data points in each category.
Step-5: Assign the new data points to that category for which the number of the neighbor is maximum.
Step-6: Our model is ready.

Then how to select the optimal K value?

There are no pre-defined statistical methods to find the most favorable value of K.
Initialize a random K value and start computing.
Choosing a small value of K leads to unstable decision boundaries.
The substantial K value is better for classification as it leads to smoothening the decision boundaries.
Derive a plot between error rate and K denoting values in a defined range. Then choose the K value as having a minimum error rate.

Ways to perform KNN
KNeighborsClassifier(n_neighbors=5, *, weights=’uniform’, algorithm=’auto’, leaf_size=30, p=2, metric=’minkowski’, metric_params=None, n_jobs=None, **kwargs)
algorithm : {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, default=’auto’
Brute Force: 
k-Dimensional Tree (kd tree):
Ball Tree: 

Comparison and Summary

Brute Force may be the most accurate method due to the consideration of all data points. Hence, no data point is assigned to a false cluster. For small data sets, Brute Force is justifiable, however, for increasing data the KD or Ball Tree is better alternatives due to their speed and efficiency.



KNN model implementation

```
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import matplotlib.ticker as ticker
%matplotlib inline
df = pd.read_csv('Telecustomers.csv')
df.head()
```
```
X = df.drop(['custcat'], axis = 1)
y = df['custcat']
from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
#Train Model and Predict
k = 4  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
Pred_y = neigh.predict(X_test)
print("Accuracy of model at K=4 is",metrics.accuracy_score(y_test, Pred_y))

error_rate = []
for i in range(1,40):
 knn = KNeighborsClassifier(n_neighbors=i)
 knn.fit(X_train,y_train)
 pred_i = knn.predict(X_test)
 error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', 
         marker='o',markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
print("Minimum error:-",min(error_rate),"at K =",error_rate.index(min(error_rate)))
```
```
acc = []
# Will take some time
from sklearn import metrics
for i in range(1,40):
    neigh = KNeighborsClassifier(n_neighbors = i).fit(X_train,y_train)
    yhat = neigh.predict(X_test)
    acc.append(metrics.accuracy_score(y_test, yhat))
    
plt.figure(figsize=(10,6))
plt.plot(range(1,40),acc,color = 'blue',linestyle='dashed', 
         marker='o',markerfacecolor='red', markersize=10)
plt.title('accuracy vs. K Value')
plt.xlabel('K')
plt.ylabel('Accuracy')
print("Maximum accuracy:-",max(acc),"at K =",acc.index(max(acc)))

```
](https://medium.com/swlh/k-nearest-neighbor-ca2593d7a3c4

## **K-Nearest Neighbor (KNN)**

K-Nearest Neighbors (KNN) is a powerful and intuitive supervised learning algorithm used for both **classification** and **regression tasks**. The fundamental idea is to predict the correct class or value for test data points by finding **K nearest training points** based on distance.

### **Key Idea**
KNN predicts the category or value by:
1. **Calculating distances** between the test data point and all training points.
2. **Selecting K closest points** (neighbors).
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

Let me know if you'd like more insights!)
