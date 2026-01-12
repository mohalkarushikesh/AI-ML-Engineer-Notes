### 🧠 What is PCA?

**Principal Component Analysis (PCA)** is a **dimensionality reduction technique** used in data analysis and machine learning. It transforms a large set of features into a smaller one that still contains most of the original information.

---

### 🎯 Why Use PCA?

- **Reduce Overfitting** in models
- **Speed up computations** with fewer features
- **Visualize high-dimensional data** in 2D or 3D
- **Remove redundancy** in correlated features

---

### ⚙️ How PCA Works (Step-by-Step)

1. **Standardize the data**  
   Center and scale all features (e.g. using `StandardScaler`)

2. **Compute the covariance matrix**  
   This represents how features vary together

3. **Compute eigenvalues and eigenvectors**  
   These help identify the directions (principal components) of maximum variance

4. **Sort components by explained variance**  
   The top components capture the most variation

5. **Project data** onto a lower-dimensional space  
   Reduce the number of features by keeping only the top principal components

---

### 📊 Key Terms

- **Principal Components**: New axes that capture the directions of max variance
- **Explained Variance Ratio**: The amount of original variance captured by each component
- **Scree Plot**: Plot of explained variance vs. number of components (look for the “elbow”)

---

### 📦 Example in Scikit-Learn

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize PCA
pca = PCA(n_components=2)  # Reduce to 2 components
X_pca = pca.fit_transform(X_scaled)

print(pca.explained_variance_ratio_)  # How much variance each PC explains
```
