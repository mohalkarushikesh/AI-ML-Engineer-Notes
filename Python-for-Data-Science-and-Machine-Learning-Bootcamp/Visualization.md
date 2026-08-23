For classification — scatter two features, color by class:

```python
import matplotlib.pyplot as plt

def viz_clf(df, x, y, target):
    plt.scatter(df[x], df[y], c=df[target], cmap="viridis", s=15)
    plt.xlabel(x); plt.ylabel(y); plt.colorbar(label=target); plt.show()
```

For regression — scatter each feature against the target:

```python
def viz_reg(df, target):
    for c in df.columns:
        if c != target:
            plt.scatter(df[c], df[target], s=10, alpha=0.5)
            plt.xlabel(c); plt.ylabel(target); plt.show()
```

Usage: `viz_clf(df, "feat1", "feat2", "label")` or `viz_reg(df, "price")`.
