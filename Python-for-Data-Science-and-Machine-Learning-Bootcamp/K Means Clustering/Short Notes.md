# **🔹 K‑Means Clustering**

## 📖 Definition
K‑Means is an **unsupervised machine learning algorithm** used for **clustering**. It partitions data into *k* clusters such that points within the same cluster are more similar to each other than to points in other clusters.

- **K‑Means → k is fixed too, but centroids update during training.**

---

## 🎯 Goal
- Group data into *k* clusters based on similarity.  
- Minimize the **within-cluster variance** (distance between points and their cluster centroid).  

---

## ⚙️ Process (Algorithm Steps)
1. **Initialize**: Choose *k* cluster centers (centroids), either randomly or using methods like *k-means++*.  
2. **Assignment Step**: Assign each data point to the nearest centroid (based on distance metric).  
3. **Update Step**: Recompute centroids as the mean of all points assigned to each cluster.  
4. **Repeat**: Continue assignment and update until centroids stabilize (convergence) or a maximum number of iterations is reached.  

---

## 📏 Distance Metric
- Most commonly: **Euclidean distance**  
- Other options: Manhattan distance, cosine similarity (depending on data type).  

---

## 📊 Objective Function
The algorithm minimizes the **total within-cluster variance**:

$$
J = \sum_{i=1}^{k} \sum_{x \in C_i} |x - \mu_i|^2
$$

Where:  
- $k$ = number of clusters  
- $C_i$ = set of points in cluster $i$  
- $x$ = a data point  
- $\mu_i$ = centroid (mean) of cluster $i$  
- $|x - \mu_i|^2$ = squared Euclidean distance  

---

<img width="561" height="283" alt="0_mQCGBdYhzZ8YMZPv" src="https://github.com/user-attachments/assets/4f05d3af-0e97-4304-a46a-f517cc8b674e" />

👉 The algorithm tries to **minimize total within-cluster variance**.

---

## ✅ Advantages
- Simple and easy to implement.  
- Fast and efficient for large datasets.  
- Scales well with the number of samples.  
- Works well when clusters are spherical and evenly sized.  

---

## ⚠️ Limitations
- Must choose *k* beforehand (not always obvious).  
- Sensitive to outliers and noisy data.  
- Assumes clusters are spherical and equally sized.  
- Different initializations can lead to different results (local minima).  

---

## 🔧 Improvements & Variants
- **k-means++** → Better initialization of centroids to improve convergence.  
- **Mini-Batch K-Means** → Faster approximation for very large datasets.  
- **Kernel K-Means** → Uses kernel methods for non-linear cluster boundaries.  

---

## 🧪 Applications
- Customer segmentation in marketing.  
- Image compression (reducing colors by clustering pixel values).  
- Document clustering in NLP.  
- Anomaly detection (points far from any cluster).  

---

# ✅ Summary
- **Type**: Unsupervised learning (clustering).  
- **Goal**: Partition data into *k* clusters by minimizing within-cluster variance.  
- **Process**: Initialize → Assign → Update → Repeat.  
- **Pros**: Simple, fast, scalable.  
- **Cons**: Requires *k*, sensitive to outliers, assumes spherical clusters.  
- **Formula**: Objective function minimizes squared distances to centroids.  

---

