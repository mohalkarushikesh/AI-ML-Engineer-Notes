Here’s a crisp set of very short notes on **K-Means Clustering**:

* **Type**: Unsupervised machine learning (clustering).
* **Goal**: Partition data into *k* clusters based on similarity.
* **Process**:

  1. Choose *k* cluster centers (centroids).
  2. Assign each point to the nearest centroid.
  3. Recompute centroids as mean of assigned points.
  4. Repeat until centroids stabilize (convergence).
* **Distance Metric**: Usually Euclidean distance.
* **Output**: *k* clusters with minimized intra-cluster variance.
* **Pros**: Simple, fast, scalable.
* **Cons**: Need to choose *k* beforehand, sensitive to outliers, assumes spherical clusters.

Want me to also make a **1-page mindmap style diagram** for quick recall?
Here’s the key **formula used in K-Means**:

**Objective Function (to minimize):**

[
J = \sum_{i=1}^{k} \sum_{x \in C_i} |x - \mu_i|^2
]

Where:

* (k) = number of clusters
* (C_i) = set of points in cluster (i)
* (x) = a data point
* (\mu_i) = centroid (mean) of cluster (i)
* (|x - \mu_i|^2) = squared Euclidean distance

👉 The algorithm tries to **minimize total within-cluster variance**.

Do you want me to also give the **centroid update formula** separately?
