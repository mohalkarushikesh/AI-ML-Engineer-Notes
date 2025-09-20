# 🧠 Feature Engineering Techniques

Feature engineering is the process of creating, scaling, and selecting the most relevant variables (features) from raw data to improve model performance.

---

## 🔧 1. Absolute Maximum Scaling

- **Description**: Rescales each feature by dividing all values by the maximum absolute value of that feature.
- **Formula**:  
  $$X_{\text{scaled}} = \frac{X_i}{\max(|X|)}$$
- **Range**: [-1, 1]
- **Sensitivity**: Highly sensitive to outliers.
- **Use Case**: When features are centered around zero but not normally distributed.

```python
import numpy as np
import pandas as pd

df = pd.read_csv('Housing.csv')
df_numeric = df.select_dtypes(include=np.number)

max_abs = np.max(np.abs(df_numeric), axis=0)
scaled_df = df_numeric / max_abs
scaled_df.head()
```

---

## 📊 2. Min-Max Scaling

- **Description**: Transforms features to a fixed range, typically [0, 1].
- **Formula**:  
  $$X_{\text{scaled}} = \frac{X - X_{\min}}{X_{\max} - X_{\min}}$$
- **Sensitivity**: Sensitive to outliers.
- **Use Case**: When features need to be bounded within a specific range.

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_numeric)
scaled_df = pd.DataFrame(scaled_data, columns=df_numeric.columns)
```

---

## 📐 3. Normalization (Vector Scaling)

- **Description**: Scales each data sample (row) so its Euclidean norm is 1.
- **Formula**:  
  $$X_{\text{scaled}} = \frac{X_i}{\|X\|}$$
- **Sensitivity**: Less sensitive to feature magnitude.
- **Use Case**: Useful in text classification, clustering, and cosine similarity-based models.

```python
from sklearn.preprocessing import Normalizer

scaler = Normalizer()
scaled_data = scaler.fit_transform(df_numeric)
scaled_df = pd.DataFrame(scaled_data, columns=df_numeric.columns)
```

---

## 📏 4. Standardization

- **Description**: Centers features by subtracting the mean and scales by standard deviation.
- **Formula**:  
  $$X_{\text{scaled}} = \frac{X_i - \mu}{\sigma}$$
- **Sensitivity**: Sensitive to outliers.
- **Use Case**: Beneficial for models assuming normal distribution (e.g., linear/logistic regression, neural networks).

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_numeric)
scaled_df = pd.DataFrame(scaled_data, columns=df_numeric.columns)
```

---

## 🛡️ 5. Robust Scaling

- **Description**: Uses median and IQR for scaling, making it robust to outliers.
- **Formula**:  
  $$X_{\text{scaled}} = \frac{X_i - \text{median}}{\text{IQR}}$$
- **Sensitivity**: Resistant to outliers.
- **Use Case**: Ideal for skewed distributions and noisy data.

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
scaled_data = scaler.fit_transform(df_numeric)
scaled_df = pd.DataFrame(scaled_data, columns=df_numeric.columns)
```

---

## 📋 Comparison Table

| Type               | Description                                      | Sensitivity to Outliers | Use Cases                                  |
|--------------------|--------------------------------------------------|--------------------------|---------------------------------------------|
| Absolute Scaling   | Divide by max absolute value                     | High                     | Centered data, not normally distributed     |
| Min-Max Scaling    | Scale to fixed range (0–1)                       | High                     | Bounded features, image data                |
| Normalization      | Scale rows to unit norm                          | Low                      | Text, clustering, cosine similarity         |
| Standardization    | Zero mean, unit variance                         | High                     | Regression, neural networks                 |
| Robust Scaling     | Median and IQR based scaling                     | Low                      | Skewed or noisy data                        |

---

## ✅ Benefits of Feature Scaling

- Improves model performance
- Enhances convergence speed
- Prevents feature bias
- Increases numerical stability
- Facilitates algorithm compatibility

---
