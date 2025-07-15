## 🧠 **What are CNNs (Convolutional Neural Networks)?**

CNNs are a class of deep neural networks primarily used for **image recognition**, **classification**, and **computer vision** tasks. They automatically and adaptively learn spatial hierarchies of features from input images.

> **Yann LeCun** is considered one of the founding fathers of deep learning and the pioneer of CNNs.

---

## 🧩 **Step-by-Step Architecture of a CNN**

---

### 🔹 **Step 1: Convolution**

Convolution is a mathematical operation applied to images using a **kernel (filter)** to extract features like edges, textures, and patterns.

📌 **Key Concepts**:
- **Input Image**: 7×7 matrix of binary values (0s and 1s)
- **Kernel (Feature Detector)**: 3×3 matrix
- **Output Feature Map**: 5×5 matrix

📷 !Convolution Setup

📷 !Feature Map Output

🧠 **Why it matters**: Helps the network learn spatial hierarchies. Early layers detect edges; deeper layers detect complex features like faces or objects.

---

### 🔧 **Common Filters (Kernels)**

| Filter Type     | Example Kernel Matrix |
|----------------|------------------------|
| **Sharpen**     | `0  0  0  0  0`<br>`0  0 -1  0  0`<br>`0 -1  5 -1  0`<br>`0  0 -1  0  0`<br>`0  0  0  0  0` |
| **Blur**        | (Typically a matrix of small positive values that average surrounding pixels) |
| **Edge Enhance**| `0  0  0`<br>`-1 -1 0`<br>`0  0  0` |
| **Edge Detect** | `0  1  0`<br>`1 -4  1`<br>`0  1  0` |
| **Emboss**      | (Highlights edges with a 3D shadow effect) |

---

### 🔹 **Step 2: Max Pooling**

Reduces the spatial dimensions of the feature map while retaining the most important information.

📌 **Types**:
- **Max Pooling**: Takes the maximum value in each patch.
- **Mean Pooling**: Takes the average.
- **Sum Pooling**: Adds up values.

🧠 **Why it matters**: Reduces computation, removes noise, and preserves features even if the image is slightly rotated or shifted.

📚 *Additional Reading*: *Evaluation for Pooling Operations in Convolutional Architectures for Object Recognition* by **Dominik Scherer et al., 2010**

---

### 🔹 **Step 3: Flattening**

Converts the 2D feature maps into a 1D vector to feed into the fully connected layers.

---

### 🔹 **Step 4: Fully Connected Layer (Dense Layer)**

Each neuron is connected to every neuron in the previous layer. This layer performs the final classification based on the features extracted.

---

## 🧪 **Activation Function: ReLU Layer**

📷 !ReLU Layer

- **ReLU (Rectified Linear Unit)** introduces non-linearity.
- **Why break linearity?** Without it, the network would behave like a linear classifier, limiting its ability to model complex patterns.

📚 *Additional Reading*:
1. *Understanding CNN with Mathematical Model* by **Jay Kuo, 2016**
2. *Delving Deep into Rectifiers* by **Kaiming He et al., 2015**

---

## 🎯 **Extra Topics**

### 🔸 **Softmax & Cross-Entropy**
- **Softmax**: Converts raw scores into probabilities.
- **Cross-Entropy**: Measures the difference between predicted and actual labels.

📚 *Additional Reading*: *Gradient-Based Learning Applied to Document Recognition* by **Yann LeCun, 1998**

---

## 🧰 **Terminology Recap**

| Term              | Description |
|-------------------|-------------|
| **Kernel / Filter** | Matrix used to extract features |
| **Stride**         | Number of steps the kernel moves |
| **Feature Map**    | Output of the convolution operation |
| **Pooling**        | Downsampling technique |
| **Flattening**     | Converts 2D to 1D |
| **Fully Connected**| Final classification layer |

---
