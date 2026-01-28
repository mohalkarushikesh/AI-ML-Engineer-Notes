## 📘 Label Propagation Algorithm (LPA)

### 🔹 Definition
- **Label Propagation** is a **semi-supervised learning algorithm** that spreads labels from a small set of labeled data points to a larger set of unlabeled points using the structure of the data (often represented as a graph).
- It assumes that **similar nodes (data points) are likely to share the same label**.

---

### 🔹 Core Idea
1. Represent data as a **graph**:
   - Nodes = data points  
   - Edges = similarity between points (e.g., distance, kernel function)  
2. Start with a few labeled nodes and many unlabeled nodes.  
3. Iteratively **propagate labels** across the graph until convergence.  

---

### 🔹 Algorithm Steps
1. **Construct similarity graph** using a kernel (e.g., Gaussian kernel).  
2. **Initialize labels**: known labels are fixed, unknown labels are empty.  
3. **Propagation rule**: each unlabeled node updates its label distribution based on neighbors.  
   - Weighted average of neighbors’ labels.  
4. **Repeat** until labels stabilize (convergence).  

---

### 🔹 Mathematical Formulation
- Let $Y$ be the label matrix (labeled + unlabeled).  
- Let $W$ be the similarity matrix (edge weights).  
- Propagation update:
```math
  Y^{(t+1)} = \alpha W Y^{(t)} + (1 - \alpha) Y^{(0)}
```
  where:
  - $\alpha$ controls the balance between propagation and initial labels.  
  - $Y^{(0)}$ is the initial label assignment.  

---

### 🔹 Applications
- **Text classification:** Spread labels across similar documents.  
- **Image recognition:** Label unlabeled images using similarity in feature space.  
- **Social networks:** Infer community membership.  
- **Biology:** Predict protein functions based on similarity networks.  

---

### 🔹 Advantages
- Works well with very few labeled examples.  
- Exploits data structure (graph-based).  
- Simple and intuitive.  

### 🔹 Limitations
- Sensitive to graph construction (choice of similarity measure).  
- May propagate incorrect labels if initial labels are noisy.  
- Computationally expensive for very large graphs.  

---

✅ **In short:** Label Propagation is a semi-supervised algorithm that spreads known labels through a similarity graph, making it powerful when labeled data is scarce but unlabeled data is abundant.

---

Here’s a **Python example of Label Propagation** using `sklearn.semi_supervised.LabelPropagation` on a toy dataset. This demonstrates how semi-supervised learning spreads labels from a few labeled points to many unlabeled ones:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.semi_supervised import LabelPropagation

# Step 1: Create a toy dataset (two moons)
X, y = datasets.make_moons(n_samples=200, noise=0.1, random_state=42)

# Step 2: Hide most labels (semi-supervised setting)
rng = np.random.RandomState(42)
y_missing = np.copy(y)
mask = rng.rand(len(y)) < 0.8   # 80% unlabeled
y_missing[mask] = -1            # -1 means unlabeled in sklearn

# Step 3: Apply Label Propagation
label_prop_model = LabelPropagation(kernel='rbf', gamma=20)
label_prop_model.fit(X, y_missing)

# Step 4: Predict labels for unlabeled data
y_pred = label_prop_model.transduction_

# Step 5: Visualize results
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap=plt.cm.Set1, s=40, edgecolor='k')
plt.title("Label Propagation on Two Moons Dataset")
plt.show()
```

---

### 🔹 What happens here:
1. **Dataset:** Two interleaving moons (classic clustering dataset).  
2. **Semi-supervised setup:** Only ~20% of points are labeled, the rest are set to `-1`.  
3. **Label Propagation:** Spreads labels across the graph using similarity (RBF kernel).  
4. **Result:** The algorithm correctly classifies most unlabeled points by leveraging structure.  

---

✅ **In short:** Label Propagation uses graph-based similarity to infer labels for unlabeled data, making it powerful when labeled data is scarce but unlabeled data is abundant.

---
