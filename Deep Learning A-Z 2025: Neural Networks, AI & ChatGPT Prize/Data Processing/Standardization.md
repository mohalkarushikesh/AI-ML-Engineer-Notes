# Standardization in Deep Learning

## 1. What is Standardization?
Standardization is a data preprocessing technique used to transform features so they have:
- Mean = 0
- Standard Deviation = 1

Formula:
```
Z = (X - mean) / std
```

## 2. Why Standardization is Important
- Helps in faster convergence during training
- Prevents features with large scales from dominating
- Improves numerical stability
- Essential for gradient-based optimization (e.g., SGD, Adam)

## 3. Difference Between Normalization and Standardization

| Aspect | Standardization | Normalization |
|--------|----------------|---------------|
| Formula | (X - mean) / std | (X - min) / (max - min) |
| Range | (-∞, +∞) | [0, 1] |
| Use Case | Gaussian-like data | Bounded data |

## 4. Standardization in Neural Networks
### a. Input Feature Standardization
- Applied before training
- Ensures all features contribute equally

### b. Batch Normalization
- Applied inside neural networks
- Normalizes layer inputs using batch statistics
- Formula:
```
BatchNorm(x) = gamma * ((x - batch_mean) / sqrt(batch_var + epsilon)) + beta
```

#### Benefits:
- Reduces internal covariate shift
- Enables higher learning rates
- Acts as regularization

## 5. Layer Normalization vs Batch Normalization

| Feature | Batch Norm | Layer Norm |
|--------|------------|-------------|
| Stats Computed | Across batch | Across features |
| Dependency | Batch size | Independent of batch |
| Use Case | CNNs | RNNs, Transformers |

## 6. Implementation Example (Python)
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

## 7. Key Points
- Always fit scaler on training data only
- Apply same transformation to validation/test sets
- Required for distance-based algorithms and neural networks

## 8. Summary
Standardization is a critical preprocessing step in deep learning that improves model performance, training stability, and convergence speed.
