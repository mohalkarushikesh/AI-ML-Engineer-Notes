## Classification and Neural Networks Concepts

### Binary vs. Multiclass Classification

- **Binary Classification**: Uses the **sigmoid** activation function to output a probability between 0 and 1.
- **Multiclass Classification**: Uses the **softmax** activation function to output a probability distribution over multiple classes.

### Loss Function: Cross-Entropy

- **Cross-entropy loss** is used to compare two probability distributions—typically the predicted distribution and the true distribution.
- It is based on the **Kullback-Leibler divergence**:
  
  \[D_{KL}(P || Q) = \sum_i P(i) \log\left(\frac{P(i)}{Q(i)}\right)\]

  where \( P \) is the true distribution and \( Q \) is the predicted distribution.

---

## Convolutional Neural Networks (CNNs)

### Convolution Operation

- A **convolution** is a mathematical operation that transforms an input matrix using a kernel (filter).
- Properties:
  - **Symmetric**
  - **Associative**
  - **Distributive**

### CNN Layer Parameters

- **Kernel Size**: Dimensions of the filter (e.g., 3×3).
- **Stride**: Number of steps the kernel moves across the input.
- **Padding**: Adds artificial borders to the input to preserve spatial dimensions.
- **Depth**: Number of filters applied, determining the depth of the output volume.

### Pooling Layers

- Used to reduce the spatial dimensions of the feature maps.
- Common types:
  - **Max Pooling**: Takes the maximum value in the receptive field.
  - **Average Pooling**: Takes the average value.

### Correlation vs. Convolution

- **Correlation** is similar to convolution but does not flip the kernel.
- If the kernel is not flipped, the operation is technically a correlation.

---

## Deteriorated Information

**Deteriorated information** refers to data or knowledge that has lost its accuracy, reliability, or relevance over time. This degradation can result from:

- **Passage of time** — Information becomes outdated or obsolete.
- **Contextual changes** — Shifts in circumstances render the information less applicable.
- **Data corruption or loss** — Due to technical issues, poor storage, or transmission errors.
- **Human error** — Mistakes in data entry, interpretation, or reporting.

While deteriorated information is generally less useful, it may still retain secondary value in certain contexts such as historical analysis, trend detection, or training robust models.

---

## Classification with TensorFlow

### Preventing Overfitting

- **Early Stopping**: Automatically stops training when validation loss stops improving.
  ```python
  from tensorflow.keras.callbacks import EarlyStopping
  early_stop = EarlyStopping(monitor='val_loss', patience=3)
  ```

- **Dropout Layers**: Randomly deactivate neurons during training to prevent overfitting.
  ```python
  from tensorflow.keras.layers import Dropout
  model.add(Dropout(0.5))
  ```

---

## Activation Functions

### ReLU (Rectified Linear Unit)

- Defined as:
  \[
  f(x) = \max(0, x)
  \]
- If the input is less than 0, output is 0; otherwise, output is the input.

---

## Training Parameters

### Batch Size

- Number of training examples used in one iteration to update model weights.
- Affects memory usage and convergence speed.

---

## Model Saving Format

### HDF5 (.h5)

- A hierarchical data format used to save entire models, including:
  - Architecture
  - Weights
  - Optimizer state

---

## Evaluation Metrics

- **Precision**: Proportion of true positives among predicted positives.
- **Recall**: Proportion of true positives among actual positives.
- **F1-Score**: Harmonic mean of precision and recall.
- **Support**: Number of actual occurrences of each class.

---

## TensorBoard

**TensorBoard** is a visualization toolkit for TensorFlow that provides:

- Tracking of metrics like loss and accuracy
- Visualization of the model graph
- Embedding projections
- Comparison of training runs

To use in Colab:
```python
%load_ext tensorboard
%tensorboard --logdir logs/
```
