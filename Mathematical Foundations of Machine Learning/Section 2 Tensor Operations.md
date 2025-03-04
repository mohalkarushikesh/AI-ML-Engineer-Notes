# Tensor and Linear Algebra Notes

## Tensor Transposition
- **Scalar Transpose**: The transpose of a scalar is itself.  
  Example: 𝑥𝑇=𝑥
- **Vector Transpose**: Converts a column vector to a row vector (or vice versa).
- **Matrix Transposition**: Flips the axes over the main diagonal (rows become columns).  
  ![Matrix Transposition](https://github.com/user-attachments/assets/a897dcd1-a14b-472a-a854-3d7a73399041)
- **Hadamard Product (Element-wise Product)**: For tensors of the same size, operations are applied element-wise by default. **Note**: This is not matrix multiplication.  
  ![Hadamard Product](https://github.com/user-attachments/assets/ee903636-78cf-4b3b-9b4d-876dadab1a0a)

---

## Basic Tensor Arithmetic

### Reduction
- **Definition**: Calculating the sum across all elements of a tensor.
  - **For a Vector x**:  
    Length n, sum all elements.
  - **For a Matrix X**:  
    Dimensions m times n:  
    - Along rows: Axis 0  
    - Along columns: Axis 1
- Reduction operations can also be applied along selected axes (e.g., max, min, mean, product).

### The Dot Product
- **Definition**: Multiply corresponding elements of two vectors (of the same length) and sum the results.  
  Example: For two vectors x and y of the same length, 𝑥.𝑦=∑𝑥𝑖𝑦𝑖
- **Significance**: The dot product is fundamental in deep learning and is performed at every artificial neuron in a deep neural network. These networks may contain millions (or more) of such neurons.

---

## Solving Linear Systems

### Method 1: Substitution
- **When to Use**: If there is a variable in the system with a coefficient of 1.
- **Steps**:
  1. Solve for one variable in terms of the other(s).
  2. Substitute back into the original equations.

### Method 2: Elimination
- **When to Use**: If no variable has a coefficient of 1.
- **Steps**:
  1. Use the addition property to eliminate a variable.
  2. If necessary, multiply one or both equations to facilitate elimination.
  3. Solve for the remaining variables.

---

