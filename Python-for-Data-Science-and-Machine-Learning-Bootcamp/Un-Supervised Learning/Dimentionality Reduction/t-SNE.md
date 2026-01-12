**t‑SNE (t‑Distributed Stochastic Neighbor Embedding) is a powerful non‑linear dimensionality reduction algorithm mainly used for visualizing high‑dimensional data in 2D or 3D. It preserves local structure (neighbors) better than linear methods like PCA, making it especially useful for clustering and pattern discovery in complex datasets.**  

---

# 📖 What is t‑SNE?
- **Type**: Unsupervised, non‑linear dimensionality reduction technique.  
- **Goal**: Map high‑dimensional data into a lower‑dimensional space (usually 2D/3D) while preserving local similarities.  
- **Use Case**: Visualization of complex datasets (images, text embeddings, genomics, etc.).

---

# 🔑 Core Intuition
- In high‑dimensional space, t‑SNE computes **pairwise similarities** between points as conditional probabilities.  
- Similar points have high probability of being neighbors; dissimilar points have low probability.  
- In low‑dimensional space, t‑SNE tries to reproduce these probabilities.  
- The algorithm minimizes the **Kullback–Leibler (KL) divergence** between the two distributions.

<img width="700" height="400" alt="image" src="https://github.com/user-attachments/assets/3489ec1d-ee7b-4efa-b1ea-b537a1ac85db" />

---

# ⚙️ Algorithm Steps
1. **Compute similarities in high‑dimensional space**  
   - Probability that point \(x_j\) is a neighbor of \(x_i\):  
   
$$
p_{j|i} = \frac{\exp(-||x_i - x_j||^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-||x_i - x_k||^2 / 2\sigma_i^2)}
$$

   - Symmetrize: $(p_{ij} = (p_{j|i} + p_{i|j}) / 2n\)$

3. **Compute similarities in low‑dimensional space**  
   - Use a Student‑t distribution (heavy tails):  
   
$$
q_{ij} = \frac{(1 + ||y_i - y_j||^2)^{-1}}{\sum_{k \neq l} (1 + ||y_k - y_l||^2)^{-1}}
$$

4. **Optimize embedding**  
   - Minimize KL divergence between \(p_{ij}\) and \(q_{ij}\):  

$$
KL(P||Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}
$$

---

# 📊 Key Features
- **Preserves local structure**: Keeps clusters and neighborhoods intact.  
- **Non‑linear**: Can separate data that PCA cannot.  
- **Heavy‑tailed distribution**: Prevents crowding problem by spreading points in low‑dimensional space.  

---

# ✅ Advantages
- Excellent for **visualizing clusters** in high‑dimensional data.  
- Handles **non‑linear relationships**.  
- Widely used in NLP (word embeddings), computer vision (image features), and bioinformatics.

---

# ⚠️ Limitations
- **Computationally expensive** for very large datasets.  
- **Non‑deterministic**: Different runs may yield slightly different embeddings.  
- **Not for downstream tasks**: Mainly for visualization, not for predictive modeling.  
- **Hyperparameter sensitivity**: Perplexity and learning rate strongly affect results.

---

# 🧪 Applications
- **Image recognition**: Visualizing CNN feature spaces.  
- **NLP**: Mapping word embeddings (Word2Vec, BERT).  
- **Genomics**: Clustering gene expression data.  
- **Anomaly detection**: Identifying outliers in complex datasets.

---

# 📝 Comparison with PCA

| Feature | PCA | t‑SNE |
|---------|-----|-------|
| Type | Linear | Non‑linear |
| Preserves | Global variance | Local neighborhoods |
| Output | Principal components | 2D/3D embedding |
| Use Case | Feature reduction | Visualization |
| Speed | Fast | Slower |

---

# ✅ Summary
- **t‑SNE** is a non‑linear dimensionality reduction algorithm specialized for **visualizing high‑dimensional data**.  
- It preserves **local structure** by modeling pairwise similarities with probabilities and minimizing KL divergence.  
- Best used for **exploratory data analysis and visualization**, not for predictive tasks.  

---


# Working 

- **Step 1 (High‑dim space)** → Compute pairwise similarities between points as probabilities (close points = high probability).  
- **Step 2 (Low‑dim space)** → Place points in 2D/3D using a Student‑t distribution to model distances.  
- **Step 3 (Optimization)** → Minimize KL divergence so that the low‑dimensional map preserves the neighborhood structure of the high‑dimensional data.  

In short: **t‑SNE converts distances into probabilities and then arranges points in 2D/3D so that neighbors stay neighbors.**  
