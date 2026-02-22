### Normalization in Machine Learning (ML) and Deep Learning (DL) is a data preprocessing technique used to scale numerical features to a similar range (typically 0 to 1 or -1 to 1). It ensures that features with larger magnitudes do not dominate the learning process and helps models converge faster during training.

### Why Normalize Data? 
- Prevent Feature Dominance: Without scaling, features with large ranges (e.g., income) can overpower features with small ranges (e.g., age), biasing models like K-Nearest Neighbors (KNN), K-Means, or Neural Networks.
- Faster Convergence: Gradient-based algorithms (linear regression, neural networks) converge faster when data is on a similar scale because the loss landscape is more spherical, reducing "bouncing".
- Numerical Stability: Normalization helps avoid "NaN" traps where extreme values in calculations exceed floating-point limits.
- Required for Specific Models: Algorithms relying on distance metrics (KNN, SVM) or gradient descent require normalization for accurate performance.

### Key Normalization Techniques 
- **Min-Max Scaling (Normalization)**: Rescales data to a fixed range, usually [0, 1]. It is sensitive to outliers.

$$\(x_{new}=\frac{x-x_{min}}{x_{max}-x_{min}}\)$$

- **Z-Score Normalization (Standardization)**: Centers data around zero with a standard deviation of 1. It is less sensitive to outliers.

$$(x_{new}=\frac{x-\mu }{\sigma }\) (where \(\mu \) is mean, \(\sigma \) is standard deviation)$$

- **Robust Scaling**: Uses the median and interquartile range (IQR), making it highly resilient to outliers.

- **Log Scaling**: Useful for highly skewed data to reduce the impact of large values.

- **Unit Vector (Vector) Normalization**: Scales a whole sample (row) to have a length (magnitude) of 1, often used for cosine similarity.

### Normalization in Deep Learning In DL, normalization is often applied to both the input data and the activation outputs of hidden layers. 

- **Batch Normalization (BatchNorm)**: Normalizes inputs to a layer across the batch dimension, mitigating internal covariate shift and speeding up training.
- **Layer Normalization (LayerNorm)**: Normalizes across all features for a single data sample, commonly used in Recurrent Neural Networks (RNNs) and Transformers.
- **Instance Normalization**: Normalizes across each channel in an image, often used in style transfer.
- **Group Normalization (GroupNorm)**: Divides features into groups and normalizes within each group, effective when batch sizes are small.

### Best Practices 

- Fit on Train, Transform on Test: Only compute normalization parameters (mean, min, max) on the training set, then apply them to the test set to prevent data leakage.
- Skip Tree-Based Models: Algorithms like Decision Trees or Random Forests are generally scale-invariant and do not require normalization.
- Handling Outliers: Use Z-score or Robust Scaling instead of Min-Max if the data contains significant outliers.
