# Dataset Visualization Cheat Sheet

## 1. First: Understand the Dataset

```text
Dataset
  │
  ├── Numerical
  │     ├── Continuous → age, salary, temperature
  │     └── Discrete   → count, number_of_purchases
  │
  ├── Categorical
  │     ├── Nominal    → city, gender, color
  │     └── Ordinal    → low, medium, high
  │
  ├── Datetime
  │     └── timestamp, date, month
  │
  └── Text
        └── reviews, comments, descriptions
```

---

# 2. Univariate Visualization

**One variable at a time**

| Data Type   | Visualization | Use                         |
| ----------- | ------------- | --------------------------- |
| Numerical   | Histogram     | Distribution                |
| Numerical   | KDE           | Smooth distribution         |
| Numerical   | Box Plot      | Spread + outliers           |
| Numerical   | Violin Plot   | Distribution + density      |
| Numerical   | ECDF          | Cumulative distribution     |
| Categorical | Bar Chart     | Category frequency          |
| Categorical | Pie Chart     | Proportion — few categories |
| Datetime    | Line Chart    | Trend over time             |

### Numerical

```text
Histogram
        █
      █ █
    █ █ █
  █ █ █ █ █
────────────────
  x → values
```

Use for:

* Distribution
* Skewness
* Central tendency
* Possible outliers

### Box Plot

```text
       ───── Maximum
          │
      ┌───┴───┐
──────│   │   │──────
      └───┬───┘
          │
       ───── Minimum

        ↑
     Outliers
```

Quickly identify:

**Median + Q1 + Q3 + IQR + Outliers**

---

# 3. Bivariate Visualization

**Two variables**

```text
             Y
             ↑
             │       •
             │    •
             │  •
             │ •
             └────────────→ X
```

| X           | Y           | Best Visualization |
| ----------- | ----------- | ------------------ |
| Numerical   | Numerical   | Scatter Plot       |
| Numerical   | Numerical   | Hexbin             |
| Categorical | Numerical   | Box Plot           |
| Categorical | Numerical   | Violin Plot        |
| Categorical | Categorical | Grouped Bar        |
| Datetime    | Numerical   | Line Plot          |
| Numerical   | Categorical | Box/Violin         |

### Scatter Plot

Use to identify:

* Correlation
* Clusters
* Outliers
* Non-linear relationships

```text
Strong +ve       Strong -ve       No correlation

   •                 • •             •   •
  •                 •                • •
 •                 •                •   •
•                 •                • •
```

---

# 4. Multivariate Visualization

**3+ variables**

| Visualization        | Best For                                      |
| -------------------- | --------------------------------------------- |
| Pair Plot            | Relationships between many numerical features |
| Correlation Heatmap  | Feature correlations                          |
| Bubble Chart         | X + Y + size                                  |
| Facet/Grid Plot      | Compare groups                                |
| Parallel Coordinates | High-dimensional data                         |
| PCA Plot             | Visualize high-dimensional data in 2D/3D      |

### Pair Plot

```text
        A       B       C

A       dist    A-B     A-C

B       B-A     dist    B-C

C       C-A     C-B     dist
```

Good for **initial EDA**.

---

# 5. Correlation Visualization

## Correlation Heatmap

```text
        A     B     C     D
A      1.0   .8   -.2    .1
B       .8   1.0  -.4    .2
C      -.2  -.4    1.0   .7
D       .1   .2     .7    1.0
```

Typical interpretation:

```text
+1  → Strong positive
 0  → No linear relationship
-1  → Strong negative
```

Use:

```python
sns.heatmap(df.corr(numeric_only=True), annot=True)
```

⚠️ **Correlation ≠ causation**

---

# 6. Categorical Data

### Frequency

```python
df["category"].value_counts().plot(kind="bar")
```

Use:

**Bar chart**

```text
A ███████████
B ███████
C ████
D ██
```

### Category vs Numerical

```python
sns.boxplot(x="category", y="value", data=df)
```

Best for comparing:

```text
Category A → distribution
Category B → distribution
Category C → distribution
```

---

# 7. Time-Series Data

```text
Value
  ↑
  │             ╭──╮
  │       ╭─────╯  ╰──╮
  │   ╭───╯            ╰─
  │───╯
  └──────────────────────→ Time
```

Use **line charts**.

Check:

* Trend
* Seasonality
* Cycles
* Spikes
* Drops
* Change points
* Missing periods

```python
df.plot(x="date", y="sales", kind="line")
```

---

# 8. Missing Values

Before modeling:

```text
Missing Data
     │
     ├── How much?
     │
     ├── Which columns?
     │
     └── Where?
```

### Visualization

```python
sns.heatmap(df.isnull(), cbar=False)
```

Or:

```python
df.isnull().sum().plot(kind="bar")
```

Useful for:

**Missingness pattern + missing percentage**

---

# 9. Outlier Detection

### Box Plot

```python
sns.boxplot(x=df["salary"])
```

### Scatter Plot

```python
sns.scatterplot(x="age", y="salary", data=df)
```

### Distribution

```python
sns.histplot(df["salary"], kde=True)
```

Look for:

```text
Normal        Skewed          Outliers

  ███          ████              ███
 █████        █████             ████
███████      ██████            █████
 █████         ███               ███
   █                         •       •
```

---

# 10. Distribution Cheat Sheet

| Shape        | What to Check        |
| ------------ | -------------------- |
| Normal       | Mean ≈ Median        |
| Right-skewed | Mean > Median        |
| Left-skewed  | Mean < Median        |
| Heavy-tailed | Extreme values       |
| Bimodal      | Possible subgroups   |
| Uniform      | Similar frequency    |
| Multimodal   | Multiple populations |

```text
Normal          Right Skew       Left Skew

    /\              /\              /\
   /  \            /  \            /  \
  /    \          /   \           /   \
 /      \________/     \_________/     \
```

---

# 11. Feature Relationships

For numerical features:

```text
Feature X ─────────────→ Feature Y
              │
              ├── Linear?
              ├── Non-linear?
              ├── Correlated?
              └── Outliers?
```

Use:

**Scatter → correlation → transformation if necessary**

---

# 12. Target Variable Analysis

For **classification**:

```python
sns.countplot(x="target", data=df)
```

Check:

```text
Class A █████████████████
Class B ███
```

⚠️ This may indicate **class imbalance**.

For **regression**:

```python
sns.histplot(df["target"], kde=True)
```

Check:

* Distribution
* Skewness
* Outliers
* Extreme values

---

# 13. EDA Visualization Flow

```text
                 DATASET
                    │
                    ▼
             ┌─────────────┐
             │ Data Types  │
             └──────┬──────┘
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
    Numerical   Categorical   Datetime
        │           │           │
        ▼           ▼           ▼
    Histogram     Bar Plot    Line Plot
    Box Plot      Count Plot
    KDE
        │
        └───────────┬───────────┘
                    ▼
              Relationships
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
       Scatter   Heatmap   Pairplot
          │
          ▼
       Outliers
       Missingness
       Patterns
          │
          ▼
       FEATURE
     ENGINEERING
```

---

# 14. Quick Decision Tree

```text
What do you want to see?
          │
          ├── Distribution?
          │      ├── Numerical → Histogram / KDE
          │      └── Category  → Bar
          │
          ├── Outliers?
          │      └── Box Plot
          │
          ├── Relationship?
          │      ├── Num + Num → Scatter
          │      ├── Cat + Num → Box / Violin
          │      └── Cat + Cat → Bar / Heatmap
          │
          ├── Correlation?
          │      └── Heatmap
          │
          ├── Time?
          │      └── Line Plot
          │
          ├── Missing values?
          │      └── Missingness Heatmap
          │
          └── Many features?
                 ├── Pair Plot
                 ├── Correlation Heatmap
                 └── PCA / t-SNE / UMAP
```

---

# 15. Python Quick Reference

```python
# Distribution
sns.histplot(df["age"], kde=True)

# Category frequency
sns.countplot(x="category", data=df)

# Box plot
sns.boxplot(x=df["salary"])

# Category vs numerical
sns.boxplot(x="department", y="salary", data=df)

# Violin
sns.violinplot(x="department", y="salary", data=df)

# Scatter
sns.scatterplot(x="age", y="salary", data=df)

# Correlation
sns.heatmap(df.corr(numeric_only=True), annot=True)

# Pair plot
sns.pairplot(df)

# Missing values
sns.heatmap(df.isnull(), cbar=False)

# Time series
sns.lineplot(x="date", y="sales", data=df)
```

## 🧠 One-line memory trick

```text
1 Variable      → Histogram / Bar / Box
2 Variables     → Scatter / Box / Bar
Many Variables  → Pairplot / Heatmap
Time            → Line
Distribution    → Histogram
Outliers        → Boxplot
Correlation     → Heatmap
Categories      → Bar
Missing Data    → Missingness Heatmap
High Dimension  → PCA / t-SNE / UMAP
```
