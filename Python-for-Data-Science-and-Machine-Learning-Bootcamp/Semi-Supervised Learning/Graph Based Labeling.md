Here are clear and structured **notes on Graph-Based Labeling** in semi-supervised learning:

---

## 📘 Graph-Based Labeling

### 🔹 Definition
- Graph-based labeling is a **semi-supervised learning approach** where data points are represented as nodes in a graph, and edges capture similarity between points.  
- Labels from a small set of labeled nodes are **propagated through the graph** to assign labels to unlabeled nodes.  
- It leverages the assumption that **similar nodes are likely to share the same label**.

---

### 🔹 Core Idea
1. **Graph Construction:**  
   - Nodes = data points.  
   - Edges = similarity (e.g., distance, kernel function).  
   - Edge weights = strength of similarity.  

2. **Label Propagation:**  
   - Known labels are fixed.  
   - Unlabeled nodes update their labels based on neighbors.  
   - Iterative propagation continues until convergence.  

3. **Objective:**  
   - Minimize label inconsistency across edges.  
   - Ensure smoothness: connected nodes should have similar labels.  

---

### 🔹 Mathematical Formulation
- Let $W$ be the similarity matrix (edge weights).  
- Let $Y$ be the label matrix.  
- Propagation update rule:
```math
  Y^{(t+1)} = \alpha W Y^{(t)} + (1 - \alpha) Y^{(0)}
```
where:  
- $\alpha$ controls balance between propagation and initial labels.  
- $Y^{(0)}$ is the initial label assignment.  

---

### 🔹 Applications
- **Text classification:** Spread labels across similar documents.  
- **Image recognition:** Label unlabeled images using feature similarity.  
- **Social networks:** Infer community membership.  
- **Biology:** Predict protein functions based on similarity networks.  

---

### 🔹 Advantages
- Works well with very few labeled examples.  
- Exploits data structure effectively.  
- Simple and intuitive.  

### 🔹 Limitations
- Sensitive to graph construction (choice of similarity measure).  
- May propagate incorrect labels if initial labels are noisy.  
- Computationally expensive for very large graphs.  

---

✅ **In short:** Graph-based labeling spreads known labels through a similarity graph, making it powerful when labeled data is scarce but unlabeled data is abundant.

---

**Python example using `sklearn.semi_supervised.LabelSpreading`** (a graph-based labeling algorithm) on a toy dataset, so you see how it works in practice. Would you like me to set that up?

Here’s a **Python example of Graph-Based Labeling** using `sklearn.semi_supervised.LabelSpreading`. This algorithm builds a similarity graph and spreads labels across it, making it a classic graph-based semi-supervised method:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.semi_supervised import LabelSpreading

# Step 1: Create a toy dataset (two moons)
X, y = datasets.make_moons(n_samples=200, noise=0.1, random_state=42)

# Step 2: Hide most labels (semi-supervised setup)
rng = np.random.RandomState(42)
y_missing = np.copy(y)
mask = rng.rand(len(y)) < 0.8   # 80% unlabeled
y_missing[mask] = -1            # -1 means unlabeled in sklearn

# Step 3: Apply Label Spreading (graph-based labeling)
label_spread_model = LabelSpreading(kernel='rbf', gamma=20, max_iter=30)
label_spread_model.fit(X, y_missing)

# Step 4: Predict labels for unlabeled data
y_pred = label_spread_model.transduction_

# Step 5: Visualize results
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap=plt.cm.Set1, s=40, edgecolor='k')
plt.title("Graph-Based Labeling (Label Spreading) on Two Moons Dataset")
plt.show()
```

---

### 🔹 What happens here:
1. **Dataset:** Two moons dataset with 200 samples.  
2. **Semi-supervised setup:** Only ~20% of points are labeled, the rest are unlabeled.  
3. **Graph construction:** Similarity graph is built using an RBF kernel.  
4. **Label Spreading:** Labels propagate across the graph until convergence.  
5. **Result:** Most unlabeled points are correctly classified by exploiting graph structure.  

---

✅ **In short:** Graph-based labeling (Label Spreading) uses the similarity graph to propagate labels smoothly across connected nodes, making it effective when labeled data is scarce but unlabeled data is abundant.

---

