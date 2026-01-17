Below are detailed, in-depth notes on the **NumPy** library in Python, a foundational package for numerical computing. These notes are structured to provide a comprehensive understanding of NumPy’s core features, functionalities, and practical applications, especially for someone familiar with programming concepts (given your earlier requests for Java DSA and Matplotlib notes). Let’s dive in!

---

### **1. Introduction to NumPy**
- **What is NumPy?**: NumPy (Numerical Python) is a library for working with arrays and performing mathematical operations efficiently. It’s the backbone of scientific computing in Python.
- **Key Features**:
  - N-dimensional array object (`ndarray`).
  - Broadcasting for element-wise operations.
  - Linear algebra, Fourier transforms, and random number generation.
- **Installation**:
  ```bash
  pip install numpy
  ```
- **Basic Import**:
  ```python
  import numpy as np
  ```

---

### **2. The NumPy Array (`ndarray`)**
- **Definition**: A multidimensional, homogeneous array of fixed-size items.
- **Creating Arrays**:
  ```python
  # From a list
  arr = np.array([1, 2, 3])  # 1D array
  arr2d = np.array([[1, 2], [3, 4]])  # 2D array

  # With built-in functions
  zeros = np.zeros((2, 3))  # 2x3 array of zeros
  ones = np.ones((3, 2))   # 3x2 array of ones
  empty = np.empty((2, 2))  # Uninitialized array
  arange = np.arange(0, 10, 2)  # Array from 0 to 9, step 2: [0, 2, 4, 6, 8]
  linspace = np.linspace(0, 1, 5)  # 5 evenly spaced numbers: [0. , 0.25, 0.5 , 0.75, 1. ]
  ```
- **Attributes**:
  - `arr.shape`: Shape of the array (e.g., `(3,)` for 1D, `(2, 2)` for 2D).
  - `arr.ndim`: Number of dimensions.
  - `arr.size`: Total number of elements.
  - `arr.dtype`: Data type (e.g., `int32`, `float64`).
  ```python
  arr = np.array([[1, 2], [3, 4]])
  print(arr.shape)  # (2, 2)
  print(arr.ndim)   # 2
  print(arr.size)   # 4
  print(arr.dtype)  # int64
  ```

---

### **3. Array Indexing and Slicing**
- **1D Array**:
  ```python
  arr = np.array([0, 1, 2, 3, 4])
  print(arr[2])     # 2
  print(arr[1:4])   # [1, 2, 3]
  print(arr[::-1])  # [4, 3, 2, 1, 0] (reverse)
  ```
- **2D Array**:
  ```python
  arr2d = np.array([[1, 2, 3], [4, 5, 6]])
  print(arr2d[0, 1])      # 2 (row 0, col 1)
  print(arr2d[:, 1])      # [2, 5] (all rows, col 1)
  print(arr2d[1, :])      # [4, 5, 6] (row 1, all cols)
  print(arr2d[0:2, 1:3])  # [[2, 3], [5, 6]]
  ```
- **Boolean Indexing**:
  ```python
  arr = np.array([1, -2, 3, -4])
  print(arr[arr > 0])  # [1, 3]
  ```

---

### **4. Array Operations**
- **Element-wise Operations**:
  ```python
  a = np.array([1, 2, 3])
  b = np.array([4, 5, 6])
  print(a + b)  # [5, 7, 9]
  print(a * b)  # [4, 10, 18]
  print(a ** 2) # [1, 4, 9]
  ```
- **Broadcasting**: Automatically applies operations to arrays of different shapes.
  ```python
  arr = np.array([[1, 2], [3, 4]])
  scalar = 2
  print(arr * scalar)  # [[2, 4], [6, 8]]
  ```
- **Universal Functions (ufuncs)**:
  ```python
  arr = np.array([0, 1, 2])
  print(np.sin(arr))  # [0. , 0.841, 0.909]
  print(np.exp(arr))  # [1. , 2.718, 7.389]
  print(np.sqrt(arr)) # [0. , 1. , 1.414]
  ```

---

### **5. Array Manipulation**
- **Reshaping**:
  ```python
  arr = np.arange(6)  # [0, 1, 2, 3, 4, 5]
  print(arr.reshape(2, 3))  # [[0, 1, 2], [3, 4, 5]]
  ```
- **Flattening**:
  ```python
  arr2d = np.array([[1, 2], [3, 4]])
  print(arr2d.flatten())  # [1, 2, 3, 4]
  ```
- **Concatenation**:
  ```python
  a = np.array([[1, 2]])
  b = np.array([[3, 4]])
  print(np.concatenate((a, b), axis=0))  # [[1, 2], [3, 4]]
  print(np.hstack((a, b)))  # горизонтально: [[1, 2, 3, 4]]
  print(np.vstack((a, b)))  # вертикально: [[1, 2], [3, 4]]
  ```
- **Transposing**:
  ```python
  arr = np.array([[1, 2], [3, 4]])
  print(arr.T)  # [[1, 3], [2, 4]]
  ```

---

### **6. Mathematical and Statistical Functions**
- **Basic Stats**:
  ```python
  arr = np.array([1, 2, 3, 4])
  print(np.mean(arr))  # 2.5
  print(np.median(arr)) # 2.5
  print(np.std(arr))   # 1.118
  print(np.sum(arr))   # 10
  print(np.min(arr))   # 1
  print(np.max(arr))   # 4
  ```
- **Axis-wise Operations**:
  ```python
  arr2d = np.array([[1, 2], [3, 4]])
  print(np.sum(arr2d, axis=0))  # [4, 6] (sum along columns)
  print(np.sum(arr2d, axis=1))  # [3, 7] (sum along rows)
  ```

---

### **7. Linear Algebra**
- **Dot Product**:
  ```python
  a = np.array([1, 2])
  b = np.array([3, 4])
  print(np.dot(a, b))  # 1*3 + 2*4 = 11
  ```
- **Matrix Multiplication**:
  ```python
  A = np.array([[1, 2], [3, 4]])
  B = np.array([[5, 6], [7, 8]])
  print(np.matmul(A, B))  # [[19, 22], [43, 50]]
  ```
- **Determinant and Inverse**:
  ```python
  from numpy.linalg import det, inv
  A = np.array([[1, 2], [3, 4]])
  print(det(A))   # -2.0
  print(inv(A))   # [[-2. ,  1. ], [ 1.5, -0.5]]
  ```

---

### **8. Random Number Generation**
- **Basic Random**:
  ```python
  print(np.random.rand(3))      # [0.1, 0.5, 0.9] (uniform 0-1)
  print(np.random.randn(3))     # [-0.2, 1.1, 0.3] (standard normal)
  print(np.random.randint(0, 10, 3))  # [4, 7, 2] (integers 0-9)
  ```
- **Seeding for Reproducibility**:
  ```python
  np.random.seed(42)
  print(np.random.rand(3))  # Always same output with seed 42
  ```
- **Shuffling**:
  ```python
  arr = np.array([1, 2, 3, 4])
  np.random.shuffle(arr)
  print(arr)  # e.g., [3, 1, 4, 2]
  ```

---

### **9. Broadcasting**
- Allows operations on arrays of different shapes by “stretching” smaller arrays.
- **Rules**:
  - Arrays must have compatible shapes (same size or one is 1).
  - Smaller array is broadcasted to match the larger one.
- **Example**:
  ```python
  a = np.array([[1, 2, 3], [4, 5, 6]])  # Shape (2, 3)
  b = np.array([1, 2, 3])              # Shape (3,)
  print(a + b)  # [[2, 4, 6], [5, 7, 9]] (b broadcasted to each row)
  ```

---

### **10. Integration with Matplotlib**
- NumPy arrays are the standard input for Matplotlib plots.
  ```python
  import matplotlib.pyplot as plt
  x = np.linspace(0, 10, 100)
  y = np.sin(x)
  plt.plot(x, y)
  plt.show()
  ```

---

### **11. Performance Tips**
- **Vectorization**: Avoid loops; use NumPy’s built-in operations.
  ```python
  # Slow (Python loop)
  arr = np.arange(1000000)
  result = [x * 2 for x in arr]
  
  # Fast (NumPy vectorized)
  result = arr * 2
  ```
- **Memory Efficiency**: Use appropriate `dtype` (e.g., `int8` vs `int64`) to save memory.

---

### **12. Advanced Features**
- **Masked Arrays** (for handling missing data):
  ```python
  from numpy import ma
  arr = np.array([1, 2, -999, 4])
  masked_arr = ma.masked_where(arr == -999, arr)
  print(masked_arr)  # [1 2 -- 4]
  ```
- **Structured Arrays** (like tables):
  ```python
  dtype = [('name', 'U10'), ('age', 'i4')]
  arr = np.array([('Alice', 25), ('Bob', 30)], dtype=dtype)
  print(arr['name'])  # ['Alice' 'Bob']
  ```

---

### **Example: Comprehensive Use Case**
```python
import numpy as np
import matplotlib.pyplot as plt

# Generate data
x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.normal(0, 0.1, 100)  # Sine wave with noise

# Compute statistics
mean_y = np.mean(y)
std_y = np.std(y)

# Plot
plt.plot(x, y, label='Data')
plt.axhline(mean_y, color='r', linestyle='--', label=f'Mean = {mean_y:.2f}')
plt.fill_between(x, mean_y - std_y, mean_y + std_y, color='gray', alpha=0.2, label='Std Dev')
plt.legend()
plt.show()
```

---

### **Key Takeaways**
- NumPy’s `ndarray` is fast and memory-efficient compared to Python lists.
- Master array creation, indexing, and vectorized operations for efficiency.
- Leverage NumPy for math, stats, and plotting prep.
- Explore advanced features like broadcasting and linear algebra for complex tasks.

Let me know if you’d like a deeper dive into any section or additional examples!
