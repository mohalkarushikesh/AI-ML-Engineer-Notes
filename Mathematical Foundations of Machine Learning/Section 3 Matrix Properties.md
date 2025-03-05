### The Frobenius Norm
- **Definition**: Measures the size of a matrix in terms of Euclidean distance.  
  It is the sum of the magnitudes of all the vectors in X.  
  ![image](https://github.com/user-attachments/assets/2d889081-85cb-4e5d-84db-a559fc7abc2d)

---

### Matrix Multiplication
- **Condition**: The number of columns in the first matrix must equal the number of rows in the second matrix.  
  ![image](https://github.com/user-attachments/assets/1da595a1-dba3-4597-b983-85b64ea58e66)

---

### Special Matrices

**1. Symmetric Matrices**:
- **Properties**:
  - Must be a square matrix.
  - Transpose of the matrix equals the matrix itself: X^T = X 

**2. Identity Matrices**:
- **Definition**: A matrix where:
  - Every element along the main diagonal is 1
  - All other elements are 0

**3. Matrix Inversion**:
- **Definition**: The inverse of a matrix is another matrix such that their product equals the identity matrix.
  - Notation: X-1
  - **Condition**: The matrix must not be singular (i.e., all columns of the matrix must be linearly independent).  
  ![image](https://github.com/user-attachments/assets/b4e7c7e2-b7f9-4fee-a5cc-928c32fb079c)

**4. Diagonal Matrices**:
- **Definition**: Only non-zero elements are along the main diagonal; all other elements are zero.  
  Example: Identity matrix.

**5. Orthogonal Matrices**: Dot-product of any two columms is zero 

### Properties of Orthogonal Matrices
- The inverse of an orthogonal matrix equals its transpose: 𝐴-1=𝐴𝑇
- The product of a matrix and its transpose is an identity matrix:  AA^T = A^T A = I
- Determinant: {det}(A) = +-1

---
