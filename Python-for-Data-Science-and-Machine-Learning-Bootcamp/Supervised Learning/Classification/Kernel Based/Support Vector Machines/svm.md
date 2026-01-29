### 🧠 **Support Vector Machine (SVM) – Quick Notes**

**1. What is SVM?**  
SVM is a supervised machine learning algorithm used mainly for **classification**, but it can also be used for regression.

**2. Basic Idea:**  
It finds the **best boundary (hyperplane)** that separates classes with the **maximum margin** — the greatest distance between data points of both classes.

**3. Key Terms:**
- **Hyperplane**: A line (in 2D), plane (in 3D), or n-dimensional boundary that separates classes.
- **Support Vectors**: Data points closest to the hyperplane; they define the margin and are critical to the model.
- **Margin**: The distance between the hyperplane and the closest support vectors from each class.

<img width="438" height="242" alt="support-vector-machine-1" src="https://github.com/user-attachments/assets/48d7ed72-e8c8-405b-a8d4-eac2b4c89a88" />

**4. Linear vs. Nonlinear SVM:**
- **Linear SVM**: Works well when data is linearly separable.
- **Nonlinear SVM**: Uses **kernel tricks** (like RBF or polynomial) to map data into higher dimensions for separation when data isn’t linearly separable.

**5. Kernels:**  
Kernels help transform data into a higher dimension:
- **Linear Kernel**
- **Polynomial Kernel**
- **Radial Basis Function (RBF) Kernel**
- **Sigmoid Kernel**

**6. Advantages:**
- Works well with high-dimensional data.
- Effective in cases where the number of features > number of samples.
- Robust to overfitting, especially with a proper kernel.

**7. Disadvantages:**
- Not ideal for very large datasets (can be slow).
- Doesn’t perform well when there’s significant noise or overlapping classes.
- Choosing the right kernel and parameters (like C and gamma) can be tricky.
