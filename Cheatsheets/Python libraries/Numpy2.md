# NumPy Cheatsheet

---

## Installation

```bash
pip install numpy

# Check version
python -c "import numpy as np; print(np.__version__)"
```

---

## Import Convention

```python
import numpy as np
```

---

## Creating Arrays

### From Python Data
```python
np.array([1, 2, 3])                        # 1D array
np.array([[1, 2, 3], [4, 5, 6]])           # 2D array
np.array([1, 2, 3], dtype=np.float32)      # Specify dtype
np.asarray(existing_list)                  # Convert without copy if possible
```

### Filled Arrays
```python
np.zeros(5)                    # [0. 0. 0. 0. 0.]
np.zeros((3, 4))               # 3×4 zeros
np.ones((3, 4))                # 3×4 ones
np.full((3, 4), 7.0)           # Fill with value
np.eye(4)                      # 4×4 identity matrix
np.empty((3, 4))               # Uninitialized (fast)

np.zeros_like(x)               # Same shape/dtype as x
np.ones_like(x)
np.full_like(x, fill_value=5)
np.empty_like(x)
```

### Ranges & Sequences
```python
np.arange(10)                  # [0, 1, ..., 9]
np.arange(0, 10, 2)            # [0, 2, 4, 6, 8]
np.arange(0.0, 1.0, 0.1)       # Float range

np.linspace(0, 1, 5)           # [0, .25, .5, .75, 1] — N evenly spaced
np.logspace(0, 3, 4)           # [1, 10, 100, 1000] — log scale
np.geomspace(1, 1000, 4)       # Geometric progression
```

### Random Arrays
```python
rng = np.random.default_rng(seed=42)   # Recommended modern API

rng.random((3, 4))             # Uniform [0, 1)
rng.standard_normal((3, 4))    # Standard normal N(0,1)
rng.normal(mean=0, scale=1, size=(3, 4))
rng.integers(0, 10, size=(3, 4))       # Random integers [0, 10)
rng.choice([1, 2, 3, 4], size=5, replace=False)
rng.shuffle(arr)               # In-place shuffle
rng.permutation(arr)           # Return shuffled copy

# Legacy (still common)
np.random.seed(42)
np.random.rand(3, 4)           # Uniform [0, 1)
np.random.randn(3, 4)          # Standard normal
np.random.randint(0, 10, (3, 4))
```

### From Files
```python
np.loadtxt('data.csv', delimiter=',', skiprows=1)
np.genfromtxt('data.csv', delimiter=',', names=True, filling_values=0)
np.load('array.npy')           # Binary .npy file
np.load('arrays.npz')          # Compressed archive → dict-like
```

---

## Array Properties

```python
x = np.random.randn(3, 4, 5)

x.shape          # (3, 4, 5)
x.ndim           # 3
x.size           # 60 (total elements)
x.dtype          # dtype('float64')
x.itemsize       # 8 (bytes per element)
x.nbytes         # 480 (total bytes)
x.T              # Transpose
x.flat           # 1D iterator over all elements
x.flags          # Memory layout info (C_CONTIGUOUS, etc.)
```

---

## Data Types

| dtype | Description | Bytes |
|-------|-------------|-------|
| `np.float64` | Default float | 8 |
| `np.float32` | Single precision | 4 |
| `np.float16` | Half precision | 2 |
| `np.int64` | Default integer | 8 |
| `np.int32` | 32-bit integer | 4 |
| `np.int16` | 16-bit integer | 2 |
| `np.uint8` | Unsigned byte (images) | 1 |
| `np.bool_` | Boolean | 1 |
| `np.complex128` | Complex number | 16 |
| `np.str_` | Unicode string | varies |

```python
x.astype(np.float32)      # Cast (returns copy)
x.astype(np.int32)
x.view(np.uint8)          # Reinterpret bytes (no copy)
```

---

## Indexing & Slicing

### Basic
```python
x = np.array([[1, 2, 3],
               [4, 5, 6],
               [7, 8, 9]])

x[0]             # First row → [1, 2, 3]
x[-1]            # Last row  → [7, 8, 9]
x[0, 1]          # Row 0, col 1 → 2
x[:, 1]          # All rows, col 1 → [2, 5, 8]
x[0:2, 1:3]      # Rows 0-1, cols 1-2
x[::2, :]        # Every other row
x[:, ::-1]       # Reverse columns
```

### Fancy Indexing
```python
x[[0, 2], :]              # Select rows 0 and 2
x[:, [0, 2]]              # Select cols 0 and 2
x[[0, 1], [1, 2]]         # Elements (0,1) and (1,2)

# Boolean mask
mask = x > 4
x[mask]                   # → [5, 6, 7, 8, 9]
x[x % 2 == 0]             # Even elements

# np.ix_ for outer indexing
rows = np.ix_([0, 2], [0, 2])  # 2×2 submatrix
```

### Advanced
```python
np.take(x, indices=[0, 2], axis=0)       # Gather along axis
np.put(x, indices=[0, 4], values=[99])   # Scatter (in-place)
np.where(x > 5, x, 0)                   # Conditional select
np.nonzero(x > 5)                        # Indices where True
np.argwhere(x > 5)                       # (N, ndim) array of indices
```

---

## Shape Manipulation

```python
x = np.random.randn(2, 3, 4)

x.reshape(6, 4)            # Reshape (returns view if possible)
x.reshape(-1, 4)           # Infer one dim automatically
x.ravel()                  # Flatten to 1D (view)
x.flatten()                # Flatten to 1D (copy)

x.squeeze()                # Remove all size-1 dims
x.squeeze(axis=0)          # Remove specific size-1 dim
x[np.newaxis, :]           # Add new axis at front
np.expand_dims(x, axis=0)  # Equivalent

x.T                        # Transpose all dims
np.transpose(x, (2, 0, 1)) # Reorder dims
x.swapaxes(0, 1)           # Swap two axes

np.moveaxis(x, source=2, destination=0)  # Move axis
```

---

## Joining & Splitting

```python
# Joining
np.concatenate([a, b, c], axis=0)    # Along existing axis
np.stack([a, b, c], axis=0)          # Along NEW axis
np.vstack([a, b])                    # Vertical stack (axis=0)
np.hstack([a, b])                    # Horizontal stack (axis=1)
np.dstack([a, b])                    # Depth stack (axis=2)
np.column_stack([a, b])              # Stack 1D as columns

# Splitting
np.split(x, 3, axis=0)              # Equal split into 3
np.split(x, [2, 4], axis=0)         # Split at indices 2 and 4
np.vsplit(x, 3)                      # Vertical split
np.hsplit(x, 3)                      # Horizontal split
np.array_split(x, 5)                 # Unequal splits allowed
```

---

## Math Operations

### Element-wise
```python
np.add(x, y)         # x + y
np.subtract(x, y)    # x - y
np.multiply(x, y)    # x * y
np.divide(x, y)      # x / y
np.power(x, 2)       # x ** 2
np.mod(x, 3)         # x % 3
np.floor_divide(x, y)# x // y
np.abs(x)
np.sqrt(x)
np.exp(x)
np.log(x)            # Natural log
np.log2(x)
np.log10(x)
np.sin(x), np.cos(x), np.tan(x)
np.ceil(x), np.floor(x), np.round(x, decimals=2)
np.sign(x)           # -1, 0, or 1
np.clip(x, a_min=0, a_max=1)
```

### Linear Algebra
```python
np.dot(a, b)              # Dot product / matrix multiply
a @ b                     # Matrix multiply (preferred)
np.matmul(a, b)           # Same as @
np.inner(a, b)            # Inner product
np.outer(a, b)            # Outer product
np.cross(a, b)            # Cross product

np.linalg.inv(A)          # Matrix inverse
np.linalg.pinv(A)         # Pseudo-inverse
np.linalg.det(A)          # Determinant
np.linalg.matrix_rank(A)  # Rank
np.linalg.norm(x)         # L2 norm
np.linalg.norm(x, ord=1)  # L1 norm
np.linalg.norm(x, ord=np.inf)  # Infinity norm
np.trace(A)               # Sum of diagonal

# Decompositions
vals, vecs = np.linalg.eig(A)     # Eigenvalues & vectors
vals, vecs = np.linalg.eigh(A)    # Symmetric matrix (faster)
U, S, Vt = np.linalg.svd(A)       # SVD
Q, R = np.linalg.qr(A)            # QR decomposition
L = np.linalg.cholesky(A)         # Cholesky decomposition

# Solving systems
x = np.linalg.solve(A, b)         # Ax = b
x, res, rank, sv = np.linalg.lstsq(A, b, rcond=None)  # Least squares
```

---

## Reductions & Statistics

```python
x = np.random.randn(3, 4)

# Basic reductions
x.sum()                      # Sum all
x.sum(axis=0)                # Sum along rows → shape (4,)
x.sum(axis=1)                # Sum along cols → shape (3,)
x.sum(axis=0, keepdims=True) # Keep dim → shape (1, 4)

x.min(), x.max()
x.min(axis=0), x.max(axis=1)
x.argmin(), x.argmax()       # Index of min/max
x.argmin(axis=0)

# Statistics
x.mean(), x.mean(axis=0)
x.std(),  x.std(axis=0, ddof=1)    # ddof=1 for sample std
x.var(),  x.var(axis=0)
np.median(x), np.median(x, axis=0)
np.percentile(x, q=75)
np.quantile(x, q=0.75)
np.cumsum(x), np.cumprod(x)
np.diff(x), np.diff(x, n=2)       # n-th differences

# Counting
np.count_nonzero(x)
np.any(x > 0)
np.all(x > 0)
np.any(x > 0, axis=0)
np.unique(x)
np.unique(x, return_counts=True)
np.bincount(int_array)            # Count occurrences of each int

# Correlation & covariance
np.corrcoef(x, y)             # Correlation matrix
np.cov(x, y)                  # Covariance matrix
np.histogram(x, bins=10)      # Histogram counts & edges
np.histogram2d(x, y, bins=10)
```

---

## Broadcasting

Rules (applied right to left):
1. If arrays have different ndim, prepend 1s to the smaller shape.
2. Dims of size 1 are stretched to match the other array.
3. Dims must be equal or one of them must be 1.

```python
# Examples
a = np.ones((3, 4))
b = np.ones((4,))          # Broadcasts to (3, 4)
c = np.ones((3, 1))        # Broadcasts to (3, 4)

# Common patterns
x = np.random.randn(100, 10)
x - x.mean(axis=0)          # Subtract per-column mean
x / x.std(axis=0)           # Divide by per-column std

# Force broadcasting dimension
col = np.array([1, 2, 3])          # (3,)
col[:, np.newaxis]                 # (3, 1) — now broadcasts over columns
```

---

## Sorting & Searching

```python
np.sort(x)                   # Return sorted copy (last axis)
np.sort(x, axis=0)           # Sort along axis
x.sort()                     # In-place sort

np.argsort(x)                # Indices that would sort x
np.argsort(x, axis=0)
x[np.argsort(x)]             # Equivalent to np.sort(x)

np.lexsort((b, a))           # Sort by a then b (last key primary)

np.partition(x, kth=3)       # Partial sort (kth smallest in place)
np.argpartition(x, kth=3)    # Indices of partial sort

np.searchsorted(sorted_arr, v)        # Binary search insertion point
np.searchsorted(sorted_arr, v, side='right')
```

---

## Set Operations

```python
np.unique(x)                        # Unique sorted values
np.union1d(a, b)                    # Union
np.intersect1d(a, b)                # Intersection
np.setdiff1d(a, b)                  # Elements in a not in b
np.setxor1d(a, b)                   # Symmetric difference
np.in1d(a, b)                       # Boolean membership test
np.isin(a, b)                       # ND-aware version of in1d
```

---

## Comparison & Logic

```python
np.equal(x, y)               # x == y element-wise
np.not_equal(x, y)           # x != y
np.greater(x, y)             # x > y
np.less_equal(x, y)          # x <= y

np.logical_and(a, b)         # Element-wise AND
np.logical_or(a, b)          # Element-wise OR
np.logical_not(a)            # Element-wise NOT
np.logical_xor(a, b)         # Element-wise XOR

np.array_equal(a, b)         # True if same shape and values
np.allclose(a, b, atol=1e-8) # True if all values close (for floats)
np.isclose(a, b)             # Element-wise close comparison
np.isnan(x), np.isinf(x), np.isfinite(x)
np.nan_to_num(x, nan=0.0)    # Replace NaN/Inf with values
```

---

## Structured Arrays & Record Arrays

```python
# Structured array (heterogeneous columns)
dt = np.dtype([('name', 'U20'), ('age', 'i4'), ('score', 'f8')])
data = np.array([('Alice', 30, 95.5), ('Bob', 25, 87.0)], dtype=dt)
data['name']    # → ['Alice', 'Bob']
data['age']     # → [30, 25]

# Record array (field access via attribute)
rec = data.view(np.recarray)
rec.name        # → ['Alice', 'Bob']
```

---

## Memory & Performance

```python
# Check if two arrays share memory
np.shares_memory(a, b)
np.may_share_memory(a, b)

# Force copy vs view
b = a.copy()             # Always a copy
b = a.view()             # Always a view (shared memory)
b = a.reshape(-1)        # View if contiguous, else copy
b = a.flatten()          # Always a copy

# Memory layout
a = np.ascontiguousarray(x)    # C-order (row-major)
a = np.asfortranarray(x)       # F-order (column-major, BLAS-friendly)
x.flags['C_CONTIGUOUS']

# Efficient operations
np.add(a, b, out=c)            # In-place with output array
np.multiply(a, b, out=a)       # Avoid allocation

# einsum — flexible multi-dim contraction
np.einsum('ij,jk->ik', A, B)   # Matrix multiply
np.einsum('ii->i', A)          # Diagonal
np.einsum('ij->i', A)          # Row sums
np.einsum('bij,bjk->bik', A, B)# Batched matmul
np.einsum('ij,ij->', A, B)     # Frobenius inner product
```

---

## Saving & Loading

```python
# Binary (fast, lossless)
np.save('array.npy', x)             # Single array
np.load('array.npy')

np.savez('arrays.npz', a=x, b=y)    # Multiple arrays (uncompressed)
np.savez_compressed('arrays.npz', a=x, b=y)  # Compressed
data = np.load('arrays.npz')
data['a'], data['b']

# Text (human-readable)
np.savetxt('data.csv', x, delimiter=',', fmt='%.6f', header='col1,col2')
np.loadtxt('data.csv', delimiter=',', skiprows=1)
```

---

## Polynomials

```python
p = np.poly1d([2, -1, 3])     # 2x² - x + 3
p(5)                           # Evaluate at x=5
p.roots                        # Roots of polynomial
p.deriv()                      # Derivative
p.integ()                      # Integral

np.polyfit(x, y, deg=2)        # Fit polynomial coefficients
np.polyval(coeffs, x)          # Evaluate polynomial
np.roots([2, -1, 3])           # Roots of 2x² - x + 3 = 0
```

---

## FFT (Signal Processing)

```python
np.fft.fft(x)                  # 1D FFT
np.fft.ifft(X)                 # Inverse FFT
np.fft.fft2(img)               # 2D FFT (images)
np.fft.fftn(x)                 # N-D FFT
np.fft.fftshift(X)             # Shift zero-freq to center
np.fft.fftfreq(n, d=1.0)       # Frequency bins
np.fft.rfft(x)                 # FFT for real input (faster)
```

---

## Useful Utilities

```python
# Printing options
np.set_printoptions(precision=4, suppress=True, linewidth=120)
np.set_printoptions(threshold=np.inf)   # Print full array

# Type checking
np.isscalar(x)
np.isreal(x), np.iscomplex(x)
isinstance(x, np.ndarray)

# Object array helpers
np.ndim(x)           # Works on lists too
np.shape(x)
np.size(x)

# Meshgrid
x = np.linspace(-1, 1, 5)
y = np.linspace(-1, 1, 5)
xx, yy = np.meshgrid(x, y)                   # (5,5) grids

# Tiling & repeating
np.tile(x, reps=(2, 3))        # Repeat whole array
np.repeat(x, repeats=3)        # Repeat each element
np.pad(x, pad_width=2, mode='constant', constant_values=0)
np.pad(x, ((1,2),(0,1)), mode='reflect')

# Diagonal operations
np.diag(x)           # Extract diagonal or make diagonal matrix
np.diag(x, k=1)      # k-th diagonal
np.diagflat(v)       # Create diagonal matrix from flat array
np.tril(x)           # Lower triangle
np.triu(x)           # Upper triangle

# Apply along axes
np.apply_along_axis(func, axis=0, arr=x)
np.vectorize(func)(x)          # Vectorize Python function (slow, convenience)
```

---

## Quick Reference: Array Creation Summary

```python
# Shape          Function
# ─────────────────────────────────
# (n,)           np.arange(n)
# (n,)           np.linspace(a, b, n)
# (m, n) zeros   np.zeros((m, n))
# (m, n) ones    np.ones((m, n))
# (n, n) eye     np.eye(n)
# (m, n) rand    rng.random((m, n))
# (m, n) normal  rng.standard_normal((m, n))
```

---

## Common Pitfalls

| Pitfall | Problem | Fix |
|---------|---------|-----|
| `a = b` | Both point to same data | `a = b.copy()` |
| `x[mask] = val` on a copy | Modifies temp, not original | Work on original array |
| Integer division | `np.array([1,2]) / np.array([3,4])` = `[0, 0]` in Python 2 | Use `from __future__ import division` or float dtype |
| Shape `(n,)` vs `(n,1)` | Broadcasting surprises | Use `reshape(-1,1)` or `[:, np.newaxis]` |
| NaN propagation | `np.mean` returns NaN if any NaN present | Use `np.nanmean`, `np.nansum`, etc. |
| `np.random.seed` not thread-safe | Race conditions in parallel code | Use `np.random.default_rng(seed)` |
| Overflow with `int32` | Large computations silently overflow | Use `int64` or check dtype first |
| Modifying slices | Slices are views — modifying affects original | Use `.copy()` if needed |

---

## NaN-safe Functions

```python
np.nansum(x)
np.nanmean(x)
np.nanstd(x)
np.nanvar(x)
np.nanmin(x),  np.nanmax(x)
np.nanargmin(x), np.nanargmax(x)
np.nanmedian(x)
np.nanpercentile(x, q=75)
np.nan_to_num(x, nan=0.0, posinf=1e9, neginf=-1e9)
```

---

*Last updated: 2025 · NumPy 2.x · For large-scale computing consider CuPy (GPU) or Dask (distributed).*
