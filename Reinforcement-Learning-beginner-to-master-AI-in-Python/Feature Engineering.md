# 🔍 Feature Engineering Process

Feature engineering is the process of transforming raw data into meaningful features that enhance the performance of machine learning models. It involves creating, transforming, extracting, selecting, and scaling features.

---

## 🛠️ 1. Creating Features

Generating new features based on domain knowledge or patterns in the data.

- **Domain-Specific**: Based on industry rules or expert knowledge (e.g., tax brackets, business logic).
- **Data-Driven**: Identifying patterns or trends directly from the data.
- **Synthetic Features**: Combining existing features (e.g., `price_per_sqft = price / area`).

---

## 🔄 2. Transforming Features

Adjusting features to improve model learning and compatibility.

- **Normalization & Scaling**: Ensures consistent feature ranges.
- **Encoding**: Converts categorical data to numerical (e.g., one-hot encoding).
- **Mathematical Transformations**: Applies log, square root, or power transformations to handle skewed data.

---

## 🧪 3. Extracting Features

Deriving meaningful features to reduce dimensionality and simplify models.

- **Dimensionality Reduction**: Techniques like PCA preserve variance while reducing feature count.
- **Aggregation & Combination**: Summing or averaging features (e.g., total income = salary + bonus).

---

## 🎯 4. Feature Selection

Choosing the most relevant subset of features to improve model performance.

### 🔍 Filter Methods
Based on statistical metrics:
- Information Gain
- Chi-Square Test
- Fisher Score
- Pearson Correlation Coefficient
- Variance Threshold
- Mean Absolute Difference
- Dispersion Ratio

### 🧪 Wrapper Methods
Based on model performance:
- Forward Selection
- Backward Elimination
- Recursive Feature Elimination (RFE)

### 🧬 Embedded Methods
Integrated into model training:
- L1 Regularization (Lasso)
- Decision Trees & Random Forests
- Gradient Boosting

---

## ⚖️ 5. Scaling Features

Ensures all features contribute equally to the model.

- **Min-Max Scaling**: Rescales values to a fixed range (e.g., 0 to 1).
- **Standard Scaling**: Normalizes features to have mean = 0 and variance = 1.

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

---

## 🔁 Steps in Feature Engineering

1. **Clean**: Handle missing values, outliers, and inconsistencies.
2. **Transform**: Normalize, encode, and mathematically adjust features.
3. **Extract**: Derive new features and reduce dimensionality.
4. **Select**: Choose the most relevant features.
5. **Iterate**: Refine and repeat based on model feedback.

---

## 🧰 Common Techniques

| Technique               | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| One-Hot Encoding        | Converts categorical variables into binary indicators.                     |
| Binning                | Transforms continuous variables into discrete bins.                         |
| Text Preprocessing     | Removes stop-words, applies stemming, and vectorizes text data.             |
| Feature Splitting      | Breaks down complex features into multiple informative components (e.g., address). |

---

## 🧪 Popular Tools for Feature Engineering

- **Featuretools**: Automated feature creation.
- **TPOT**: AutoML tool for pipeline optimization.
- **DataRobot**: Enterprise AI platform with feature engineering capabilities.
- **AlteryX**: Data preparation and analytics platform.
- **H2O.ai**: Open-source machine learning and feature engineering toolkit.

---

### 💡 Don't Forget

Feature engineering is often the most critical step in building high-performing models. It requires creativity, domain knowledge, and iterative refinement to uncover the most predictive signals in your data.

--- 

## ✅ Benefits of Feature Scaling

- Improves model performance
- Enhances convergence speed
- Prevents feature bias
- Increases numerical stability
- Facilitates algorithm compatibility

---
