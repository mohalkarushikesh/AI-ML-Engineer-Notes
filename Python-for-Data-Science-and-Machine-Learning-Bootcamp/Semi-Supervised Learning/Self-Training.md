## 📘 Self-Training Algorithm

### 🔹 Definition
- **Self-Training** is a **wrapper method** for semi-supervised learning.  
- It starts with a small set of labeled data and a large set of unlabeled data.  
- A base classifier is trained on the labeled data, then iteratively labels the unlabeled data with high-confidence predictions, adding them to the training set.

---

### 🔹 Core Idea
- Use the model’s own predictions to **bootstrap** the training process.  
- Gradually expand the labeled dataset by trusting the classifier’s confident predictions.  

---

### 🔹 Algorithm Steps
1. **Train initial classifier** on the labeled dataset.  
2. **Predict labels** for unlabeled data.  
3. **Select high-confidence predictions** (above a threshold).  
4. **Add these pseudo-labeled samples** to the labeled dataset.  
5. **Retrain classifier** with the expanded dataset.  
6. **Repeat** until no more confident predictions can be added or convergence is reached.  

---

### 🔹 Applications
- **Text classification:** Labeling documents with few initial labels.  
- **Image recognition:** Expanding training sets when manual labeling is expensive.  
- **Medical data:** Semi-supervised learning with limited annotated samples.  
- **Speech recognition:** Leveraging large unlabeled audio corpora.  

---

### 🔹 Advantages
- Simple and easy to implement.  
- Can significantly improve performance when labeled data is scarce.  
- Works with any base classifier (SVM, decision tree, neural network, etc.).  

### 🔹 Limitations
- Risk of **error propagation**: incorrect pseudo-labels can reinforce mistakes.  
- Sensitive to confidence threshold selection.  
- Works best when the base classifier is strong and initial labeled data is representative.  

---

### 🔹 Example (Conceptual)
- Start with 100 labeled images of cats and dogs, plus 10,000 unlabeled images.  
- Train a classifier on the 100 labeled samples.  
- Predict labels for the 10,000 unlabeled images.  
- Add the 500 most confident predictions to the labeled set.  
- Retrain the classifier and repeat.  

---

✅ **In short:** Self-Training is a semi-supervised learning technique that iteratively expands the labeled dataset using confident predictions from a base classifier, making it useful when labeled data is scarce but unlabeled data is abundant.


---

Here’s a **Python example of Self-Training** using `sklearn.semi_supervised.SelfTrainingClassifier` on a toy dataset. This shows how a base classifier can iteratively expand its training set with pseudo-labeled data:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.tree import DecisionTreeClassifier

# Step 1: Create a toy dataset (two moons)
X, y = datasets.make_moons(n_samples=200, noise=0.2, random_state=42)

# Step 2: Hide most labels (semi-supervised setup)
rng = np.random.RandomState(42)
y_missing = np.copy(y)
mask = rng.rand(len(y)) < 0.8   # 80% unlabeled
y_missing[mask] = -1            # -1 means unlabeled in sklearn

# Step 3: Define base classifier (Decision Tree here)
base_clf = DecisionTreeClassifier(max_depth=3, random_state=42)

# Step 4: Wrap with Self-Training
self_training_model = SelfTrainingClassifier(base_clf, criterion='k_best', k_best=50)
self_training_model.fit(X, y_missing)

# Step 5: Predict labels for all data
y_pred = self_training_model.predict(X)

# Step 6: Visualize results
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap=plt.cm.Set1, s=40, edgecolor='k')
plt.title("Self-Training on Two Moons Dataset")
plt.show()
```

---

### 🔹 What happens here:
1. **Dataset:** Two moons dataset with 200 samples.  
2. **Semi-supervised setup:** Only ~20% of points are labeled, the rest are set to `-1`.  
3. **Base classifier:** A decision tree is used as the learner.  
4. **Self-Training:** The model iteratively adds pseudo-labeled samples to the training set.  
5. **Result:** The classifier learns from both labeled and confidently pseudo-labeled data, improving accuracy.  

---

✅ **In short:** Self-Training leverages a base classifier to expand the labeled dataset with confident predictions, making it a simple yet effective semi-supervised learning technique.

---
