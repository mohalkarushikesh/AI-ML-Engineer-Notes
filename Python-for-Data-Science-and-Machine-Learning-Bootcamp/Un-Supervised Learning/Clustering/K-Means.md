### 🌟 **K-Means Clustering (Unsupervised Learning)**

**Purpose**: Automatically group similar data points into clusters without pre-labeled outcomes.

---

### 🔧 **How It Works**

1. **Choose the number of clusters**:  
   Decide on *K*, the number of clusters you want the algorithm to find.

2. **Initialize**:  
   Randomly assign each point to one of the K clusters OR initialize K random centroids.

3. **Repeat until convergence**:
   - **Step 1**: For each cluster, compute the **centroid** (mean of the points in that cluster).
   - **Step 2**: Reassign each point to the nearest centroid.

4. **Convergence** happens when assignments no longer change or changes fall below a threshold.

---

### 📐 **Choosing the Right K: The Elbow Method**

The Elbow Method helps determine the optimal number of clusters by plotting **inertia** (sum of squared distances from each point to its cluster's centroid) as a function of K.

#### Steps:
1. Run K-means for a range of K values (e.g. 1–10).
2. Plot **K vs Inertia**.
3. Look for the "elbow"—the point at which adding more clusters doesn’t significantly reduce inertia.  
   👉 This elbow is typically a good value for K.

#### Example using Scikit-Learn & Matplotlib:

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

inertia = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)  # X is your dataset
    inertia.append(kmeans.inertia_)

plt.plot(k_range, inertia, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal K')
plt.show()
```

---

### 💡 Extra Notes:

- **K-Means is sensitive to scale** → Always scale your data (e.g. with StandardScaler).
- **Random initialization** can impact results → Use `n_init` and `random_state` for stability.
- **K-Means tries to minimize intra-cluster variance**, but it assumes spherical clusters and equal density.
