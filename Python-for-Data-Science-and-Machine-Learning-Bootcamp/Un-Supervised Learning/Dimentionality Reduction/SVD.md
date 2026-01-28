## 📘 Singular Value Decomposition (SVD)

### 🔹 Definition
For any real matrix $A$ of size $m \times n$:

$$A = U \Sigma V^T$$

- $U$: orthogonal matrix ($m \times m$) → left singular vectors  
- $\Sigma$: diagonal matrix ($m \times n$) → singular values  
- $V^T$: transpose of orthogonal matrix ($n \times n$) → right singular vectors  

---

### 🔹 Key Properties
- Singular values in $\Sigma$ are non-negative and sorted in descending order.  
- Rank of $A$ = number of non-zero singular values.  
- SVD generalizes eigen-decomposition for non-square matrices.  

---

### 🔹 Applications in AI/ML
1. **Dimensionality Reduction**  
   - Similar to PCA, SVD reduces data dimensions while preserving variance.  
   - Used in latent semantic analysis (LSA) for text data.  

2. **Noise Reduction**  
   - Approximate matrix with top $k$ singular values → filters out noise.  

3. **Recommendation Systems**  
   - Matrix factorization (Netflix Prize problem) uses SVD to predict missing ratings.  

4. **Image Compression**  
   - Represent images with fewer singular values → smaller storage, faster transmission.  

5. **Natural Language Processing**  
   - Word embeddings and topic modeling (LSA) rely on SVD.  

---

### 🔹 Example
Suppose we have a $3 \times 2$ matrix:

$$A = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}$$

SVD decomposes it into:

$$A = U \Sigma V^T$$

where:
- $U$ contains orthogonal basis for column space.  
- $\Sigma$ contains singular values.  
- $V^T$ contains orthogonal basis for row space.  

---

### 🔹 Comparison with PCA
| PCA | SVD |
|-----|-----|
| Eigen-decomposition of covariance matrix | Factorization of original matrix |
| Requires square covariance matrix | Works for any rectangular matrix |
| Used for variance maximization | Used for general decomposition |

---

✅ **In short:** SVD breaks a matrix into orthogonal components, revealing structure and enabling dimensionality reduction, compression, and latent factor discovery — making it a backbone of modern ML techniques.

---

Here’s a **Python example using SVD** with `numpy.linalg.svd` to show how it decomposes a matrix and can be used for dimensionality reduction or compression:

```python
import numpy as np

# Step 1: Create a sample matrix
A = np.array([[1, 0],
              [0, 1],
              [1, 1]])

print("Original Matrix A:")
print(A)

# Step 2: Perform SVD
U, Sigma, VT = np.linalg.svd(A)

print("\nU (Left singular vectors):")
print(U)

print("\nSigma (Singular values):")
print(Sigma)

print("\nV^T (Right singular vectors):")
print(VT)

# Step 3: Reconstruct A using SVD
Sigma_matrix = np.zeros((U.shape[0], VT.shape[0]))
np.fill_diagonal(Sigma_matrix, Sigma)
A_reconstructed = U @ Sigma_matrix @ VT

print("\nReconstructed Matrix A:")
print(A_reconstructed)
```

---

### 🔹 What happens here:
1. **Matrix \(A\):** A simple $(3 \times 2\)$ matrix.  
2. **SVD Decomposition:** Breaks $(A\) into \(U\)$ , $(\Sigma\)$ , and $(V^T\)$ .  
3. **Singular Values:** Show the importance (strength) of each component.  
4. **Reconstruction:** Multiplying back $(U \Sigma V^T\)$ gives the original matrix.  

---

✅ This demonstrates how SVD decomposes a matrix into orthogonal components and how you can reconstruct it. In practice, you can keep only the top \(k\) singular values to approximate \(A\), which is the basis for **dimensionality reduction and compression**.


**Image compression example** (using SVD to reduce the size of an image while keeping most of its structure)?
