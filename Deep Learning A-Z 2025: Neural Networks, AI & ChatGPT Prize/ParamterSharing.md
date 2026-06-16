```markdown
# Parameter Sharing (Simple Explanation)

**Parameter sharing** is a technique where the same set of weights (parameters) is used across multiple parts of a model instead of having separate weights for each part.

## ✅ Why it is used
- Reduces the number of parameters
- Saves memory
- Helps prevent overfitting
- Improves generalization

## ✅ Example
In **Convolutional Neural Networks (CNNs)**:
- A single filter (kernel) is applied across the entire image
- The same weights are reused for every location

👉 Instead of learning new weights for each pixel, the model **shares one filter across all positions**

## ✅ Key Idea
> Learn once, reuse everywhere

## ✅ Summary
Parameter sharing makes models:
- Efficient
- Faster to train
- Better at recognizing patterns (like edges in images)

---
```
