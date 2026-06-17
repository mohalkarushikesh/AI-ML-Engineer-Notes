# Quasi-Recurrent Neural Network (QRNN)

A **Quasi-Recurrent Neural Network (QRNN)** is a sequence modeling architecture that alternates between **convolutional layers** (applied across time) and a minimalist **recurrent pooling function** (applied across feature channels). It achieves the predictive accuracy of a traditional LSTM while processing data up to **16× faster** due to parallel computation.

## Why QRNNs Work

Traditional RNNs and LSTMs compute states sequentially, creating a bottleneck that prevents parallelization across time. QRNNs divide sequence modeling into two independent steps:

### 1. Convolutional Step (Across Time)

The model processes the sequence using standard **1D masked convolutions**. These operate across multiple timesteps in parallel (without depending on the previous timestep's output), generating:

- Candidate hidden states
- Forget gates
- Output gates

### 2. Recurrent Step (Across Channels)

The model applies element-wise matrix multiplications to combine the convolutions. This operation:

- Is applied sequentially across time.
- Operates entirely in parallel across feature channels.
- Avoids expensive fully connected recurrent matrix multiplications.

---

# Core Benefits

### • High Parallelism

Because the recurrent state is computed without inter-channel dependency, it can be parallelized, leading to **2× to 17× faster training and inference** compared to highly optimized LSTM implementations.

### • Better Accuracy

Research has shown that stacked QRNNs often yield better predictive accuracy than standard LSTMs with the same hidden size.

### • Interpretability

Since the feature channels remain independent during the recurrent pooling step, the internal states are more interpretable than those of standard RNNs.

---

# Where to Find Implementations

## • PyTorch

You can explore the official **Salesforce PyTorch QRNN Repository** for optimized CUDA implementations.

## • Research Paper

For a deeper dive into the mathematical foundation and experimental results, read the original paper:

**"Quasi-Recurrent Neural Networks" (arXiv:1611.01576)**  
**Authors:** James Bradbury et al.

---

# Further Exploration

If you'd like, you can explore:

- **Mathematical formulations and equations** for pooling variants:
  - f-pooling
  - fo-pooling
  - ifo-pooling

- **Comparisons with other sequence models**, such as:
  - Transformers
  - GRUs
  - LSTMs

- **Implementation guides** for integrating QRNNs into:
  - PyTorch projects
  - TensorFlow projects

---

Let me know what you would like to explore next.
