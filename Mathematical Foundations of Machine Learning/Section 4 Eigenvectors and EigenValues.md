# 📘 Linear Algebra 2: Matrix Applications

## 🎯 Session Goal
- Use **tensors in Python** to solve systems of equations.  
- Identify **meaningful patterns in data** using matrix operations.  

---

## 🔹 Review: Introductory Linear Algebra
1. **Linear Algebra** is a branch of mathematics that solves **linear equations**, dealing with **vectors, matrices, eigenvalues**, and **eigenvectors**, essential in computational fields.

---

## 🔹 Applications of Linear Algebra
### 2.1 Machine Learning
- Supports optimization, **matrix operations**, and high-dimensional data manipulation.  

### 2.2 Dimensionality Reduction
- **PCA** transforms data into a lower-dimensional space.  

### 2.3 Ranking
- **Eigenvector calculations** (e.g., **Google’s PageRank**) determine importance in search results.  

### 2.4 Recommender Systems
- **SVD** identifies user preference patterns for personalized recommendations.  

### 2.5 Natural Language Processing (NLP)
- **Topic Modeling (LSA):** extracts hidden topics.  
- **Semantic Analysis:** matrix factorization improves language understanding.  

---

## 🔹 Matrix Inversion
- Avoids **overdetermination**, **underdetermination**, **no solutions**, and **infinite solutions**.  

---

## 🔹 Eigen Decomposition
1. Applying matrices  
2. Affine transformations  

---

## 🔹 Affine Transformations
Affine transformations preserve points, straight lines, and planes. They maintain relative proportions of figures, though angles and lengths might not always stay the same.  
👉 In simple terms: affine transformations ensure **“straightness” and “parallelism”** are preserved.

### Types of Affine Transformations
1. **Translation** → Shifting an object from one location to another.  
2. **Scaling** → Enlarging or shrinking an object proportionally.  
3. **Rotation** → Rotating around a fixed point or axis.  
4. **Reflection** → Flipping over a line or plane (mirror image).  
5. **Shear** → Skewing in one direction (rectangle → parallelogram).  

### Mathematical Representation
Affine transformations can be represented using matrices. In 2D, it often looks like this:

![image](https://github.com/user-attachments/assets/7397e3eb-00b5-4206-bb8e-ab94d9876219)

- $(a, b, c, d)$ → scaling, rotation, shear.  
- $(e, f)$ → translation (shift).  

In 3D → use **4×4 matrices**, including $z$-axis operations.  

### Applications
- **Computer Graphics** → scaling, rotating, translating images/3D models.  
- **Image Processing** → aligning or transforming images.  
- **Robotics** → mapping movements in space.  
- **Machine Learning** → data transformations in algorithms.  

👉 Check blog post: *Affine transformations in Python* (apply on images + vectors).  

---

## 🔹 Eigenvectors & Eigenvalues
- **Eigenvector** → a vector that doesn’t change direction during a transformation, only magnitude (scaled by eigenvalue).  
  - German “eigen” → “characteristic vector.”  
  - Example cases:  
    - **Flipping matrix** → red & blue vectors are eigenvectors.  
    - **Shearing matrix** → blue vector is eigenvector with eigenvalue = 1.  
    - Eigenvalues can be **negative** (same eigenvector, eigenvalue = -1).  

- **Eigenvalue** → scalar showing how much a vector is stretched/squished.  
  - Scales eigenvector $v$.  
  - Derived algebraically (QR algorithm, 1950s by Vera Kublanovskaya & John Francis).  
  - In practice → use NumPy `eig()` method.  

**NumPy eig() returns:**  
- Vector of eigenvalues.  
- Matrix of eigenvectors.  

![image](https://github.com/user-attachments/assets/8b35dd4c-0fda-46eb-a9a2-24d6d58d264c)

![{C8E1DBA3-154D-4BA9-8983-ECA8D7337540}](https://github.com/user-attachments/assets/4dc954d5-99a4-4978-a347-55aa11fb4da7)

---

## 🔹 Matrix Determinants
- Map **square matrix → scalar**.  
- Determine if matrix can be inverted.  
- If $\det(X) = 0$ → inverse $X^{-1}$ can’t be computed (no solution or infinite solutions).  

### Determinant of 2×2 Matrix
\[
|X| = ad - bc
\]  
For matrix:  
\[
\begin{bmatrix} a & b \\ c & d \end{bmatrix}
\]

---

## 🔹 Determinants of Larger Matrices
- Generalized using **recursion**.  

---

## 🔹 Determinants and Eigenvalues
- $\det(X)$ = product of all eigenvalues of $X$.  
- Relationship: if any eigenvalue = 0 → determinant = 0.  

---

## 🔹 Eigen Decomposition
- Formula: $X = PDP^{-1}$ (where $D$ = diagonal matrix of eigenvalues, $P$ = eigenvectors).  

### Applications of Eigen Decomposition
- Dimensionality reduction (PCA).  
- Data compression.  
- Identifying latent patterns in data.  
- Ranking & recommendation systems.  

---
