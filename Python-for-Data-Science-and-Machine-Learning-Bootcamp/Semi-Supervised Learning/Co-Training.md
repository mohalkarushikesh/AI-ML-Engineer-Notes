## 📘 Co-Training Algorithm

### 🔹 Definition
- **Co-Training** is a semi-supervised learning technique where **two classifiers** are trained on different, complementary views of the same dataset.  
- Each classifier labels unlabeled data, and the most confident predictions are added to the labeled set of the other classifier.  
- It leverages the idea that different feature sets can provide independent and complementary information.

---

### 🔹 Core Idea
- If data can be split into two **conditionally independent feature sets** (views), then two classifiers can help each other learn.  
- Example: For a webpage classification task:  
  - View 1: Words on the page.  
  - View 2: Anchor text of links pointing to the page.  

---

### 🔹 Algorithm Steps
1. **Split features** into two distinct views.  
2. **Train two classifiers** on the labeled data (one per view).  
3. **Predict labels** for unlabeled data.  
4. **Select high-confidence predictions** from each classifier.  
5. **Add these pseudo-labeled samples** to the labeled set of the other classifier.  
6. **Retrain classifiers** with expanded labeled sets.  
7. **Repeat** until convergence or no more confident predictions.  

---

### 🔹 Applications
- **Webpage classification:** Using page content and link text.  
- **Speech recognition:** Using audio features and text transcripts.  
- **Image classification:** Using visual features and metadata.  
- **Natural language processing:** Using different linguistic feature sets (syntax vs. semantics).  

---

### 🔹 Advantages
- Exploits multiple views of data for better learning.  
- Can significantly improve accuracy with few labeled examples.  
- Reduces risk of overfitting compared to single-view self-training.  

### 🔹 Limitations
- Requires data with **natural multiple views** (not always available).  
- Assumes conditional independence between views, which may not hold in practice.  
- Error propagation possible if classifiers reinforce incorrect labels.  

---

### 🔹 Example (Conceptual)
- Task: Classify emails as spam or not spam.  
- View 1: Email body text.  
- View 2: Email metadata (subject line, sender info).  
- Two classifiers train separately, then exchange confident predictions to expand the labeled dataset.  

---

✅ **In short:** Co-Training is a semi-supervised algorithm that uses two classifiers trained on different views of the data to iteratively label unlabeled samples, making it powerful when datasets naturally have multiple feature sets.

---

Here’s a **Python example of Co-Training** using two classifiers on a toy dataset. Since scikit-learn doesn’t have a built-in `CoTrainingClassifier`, we’ll implement a simple version to demonstrate the concept:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# Step 1: Create a toy dataset (two moons)
X, y = datasets.make_moons(n_samples=200, noise=0.2, random_state=42)

# Step 2: Hide most labels (semi-supervised setup)
rng = np.random.RandomState(42)
y_missing = np.copy(y)
mask = rng.rand(len(y)) < 0.8   # 80% unlabeled
y_missing[mask] = -1            # -1 means unlabeled

# Step 3: Split features into two "views"
# (Here we artificially split into halves for demonstration)
X_view1 = X[:, [0]]   # first feature
X_view2 = X[:, [1]]   # second feature

# Step 4: Initialize two classifiers
clf1 = GaussianNB()
clf2 = DecisionTreeClassifier(max_depth=3, random_state=42)

# Train on labeled data only
labeled_mask = y_missing != -1
clf1.fit(X_view1[labeled_mask], y[labeled_mask])
clf2.fit(X_view2[labeled_mask], y[labeled_mask])

# Step 5: Co-Training loop (simplified)
for iteration in range(5):
    # Predict unlabeled data
    unlabeled_mask = y_missing == -1
    
    # Classifier 1 predicts labels for unlabeled data
    pred1 = clf1.predict(X_view1[unlabeled_mask])
    # Classifier 2 predicts labels for unlabeled data
    pred2 = clf2.predict(X_view2[unlabeled_mask])
    
    # Select a few confident predictions (here we just take first 5 for demo)
    indices = np.where(unlabeled_mask)[0][:5]
    
    # Add predictions from clf1 to clf2’s training set
    clf2.fit(np.vstack([X_view2[labeled_mask], X_view2[indices]]),
             np.hstack([y[labeled_mask], pred1[:5]]))
    
    # Add predictions from clf2 to clf1’s training set
    clf1.fit(np.vstack([X_view1[labeled_mask], X_view1[indices]]),
             np.hstack([y[labeled_mask], pred2[:5]]))
    
    # Update mask (mark these as labeled now)
    y_missing[indices] = pred1[:5]

# Step 6: Final predictions
final_pred1 = clf1.predict(X_view1)
final_pred2 = clf2.predict(X_view2)

# Combine predictions (majority vote)
final_pred = np.where(final_pred1 == final_pred2, final_pred1, final_pred2)

# Step 7: Visualize results
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=final_pred, cmap=plt.cm.Set1, s=40, edgecolor='k')
plt.title("Co-Training on Two Moons Dataset")
plt.show()
```

---

### 🔹 What happens here:
1. **Dataset:** Two moons dataset with 200 samples.  
2. **Semi-supervised setup:** Only ~20% of points are labeled, the rest are unlabeled.  
3. **Two views:** Features are split into two subsets (for demonstration).  
4. **Two classifiers:** Naive Bayes and Decision Tree are trained separately.  
5. **Co-Training loop:** Each classifier labels some unlabeled data and shares it with the other.  
6. **Final prediction:** Combine outputs (majority vote).  
7. **Result:** Labels propagate through collaboration, improving classification accuracy.  

---

✅ **In short:** Co-Training leverages multiple views of data by letting classifiers teach each other, making it effective when datasets naturally have complementary feature sets.

---
