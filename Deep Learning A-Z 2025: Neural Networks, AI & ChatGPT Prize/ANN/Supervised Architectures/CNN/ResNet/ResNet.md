Below is a **corrected, structured, and deeper version of your ResNet notes**.
I have **NOT removed any images** and I’ve **kept your structure**, only **fixed inaccuracies, added missing theory, clarified math, and improved flow**.

---

## 📘 ResNet Architecture (2015, Microsoft Research)

**Paper:** *Deep Residual Learning for Image Recognition*
**Authors:** Kaiming He et al.
**Conference:** CVPR 2016 (won ImageNet 2015)

---

## 🔹 Core Motivation (Why ResNet?)

Before ResNet, simply **stacking more layers** caused:

* **Vanishing / exploding gradients**
* **Degradation problem** → deeper networks performed *worse* than shallower ones (even on training data)

❌ Not overfitting
❌ Not lack of data
✅ Optimization difficulty

**ResNet solved this using residual (skip) connections.**

---

## 🔹 Key Idea (Residual Learning)

Instead of learning a **direct mapping**:

[
H(x)
]

ResNet forces the network to learn a **residual**:

[
F(x) = H(x) - x
]

So the original function becomes:

[
H(x) = F(x) + x
]

➡️ If identity mapping is optimal, the network can simply learn
[
F(x) = 0
]

This makes **deep networks much easier to optimize**.

---

## 🔹 High-Level Architecture (ImageNet)

### 1️⃣ Input

* Image size: **224 × 224 × 3**

---

### 2️⃣ Initial Convolution + Pooling

* **7×7 convolution**, stride 2, 64 channels
* **Batch Normalization + ReLU**
* **3×3 Max Pooling**, stride 2

This aggressively reduces spatial size early.

---

### 3️⃣ Residual Blocks (Core of ResNet)

Each block follows:

[
y = F(x, W) + x
]

Where:

* (x) → input (identity shortcut)
* (F(x, W)) → stacked convolutions
* Addition happens **element-wise**

---

### 🔹 Types of Residual Blocks

#### 🟦 A. Basic Block (ResNet-18, ResNet-34)

Structure:

```
3×3 Conv → BN → ReLU
3×3 Conv → BN
+ Identity
→ ReLU
```

Used for **shallower ResNets**.

---

#### 🟩 B. Bottleneck Block (ResNet-50, 101, 152)

Structure:

```
1×1 Conv (reduce channels)
3×3 Conv
1×1 Conv (restore channels)
+ Shortcut
→ ReLU
```

Purpose:

* Reduce computation
* Enable very deep networks (100+ layers)

---

### 📌 Bottleneck intuition

* 1×1 conv reduces dimensionality → cheap
* 3×3 conv does spatial learning
* 1×1 conv expands back

---

### 4️⃣ Stacking Residual Layers

| Model      | Blocks     | Total Layers |
| ---------- | ---------- | ------------ |
| ResNet-18  | Basic      | 18           |
| ResNet-34  | Basic      | 34           |
| ResNet-50  | Bottleneck | 50           |
| ResNet-101 | Bottleneck | 101          |
| ResNet-152 | Bottleneck | 152          |

Each stage:

* Reduces spatial size
* Increases channels (64 → 128 → 256 → 512)

---

### 5️⃣ Handling Dimension Mismatch (Important 🔥)

When dimensions of (x) and (F(x)) differ:

#### ✅ Option 1: Zero Padding

* Pad identity with zeros
* No extra parameters
* Used in smaller networks

#### ✅ Option 2: Linear Projection (Most common)

[
y = F(x) + W_s x
]

* (W_s): 1×1 convolution
* Matches channels and spatial size
* Used when stride ≠ 1

---

### 6️⃣ Global Average Pooling (GAP)

* Replaces large fully connected layers
* Converts feature map → single value per channel
* Reduces parameters & overfitting

---

### 7️⃣ Fully Connected + Softmax

* Dense layer → **1000 ImageNet classes**
* Softmax for classification

---

## 🔹 Residual Connection Visuals (UNCHANGED)

### 1 Residual Block

![skip\_connection](https://github.com/user-attachments/assets/ab669a97-6ea7-4c7a-b4fb-a0939b77bbf1)

### ResNet-34 Structure

<img width="800" height="450" alt="The-structure-of-the-ResNet34-CNN-Network-The-input-of-the-network-is-the-preprocessed" src="https://github.com/user-attachments/assets/fe65dadf-4d8a-43b4-ad90-f1c3c9dc4712" />

### Overall ResNet Architecture

<img width="1123" height="487" alt="ResNet" src="https://github.com/user-attachments/assets/25683087-5883-44ff-875c-100c434c575e" />

---

## 🔹 Math Behind Why ResNet Works

### Gradient Flow

For residual layer:
[
y = F(x) + x
]

Gradient:
[
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \left(1 + \frac{\partial F}{\partial x}\right)
]

➡️ Even if ( \frac{\partial F}{\partial x} \approx 0 ),
gradient can still flow via **identity path**.

✅ Solves vanishing gradient
✅ Enables deep backpropagation

---

## 🔹 Problems ResNet Solves

### 1️⃣ Vanishing / Exploding Gradient

* Identity shortcuts provide direct gradient paths

### 2️⃣ Degradation Problem

* Deeper model ≠ worse training accuracy
* Residuals prevent performance collapse

### 3️⃣ Optimization Difficulty

* Easier loss surface
* Faster convergence

---

## 🔹 Parameter Counts (Correct & Verified)

| Model      | Parameters |
| ---------- | ---------- |
| ResNet-18  | ~11.7M     |
| ResNet-34  | ~21.8M     |
| ResNet-50  | ~25.6M     |
| ResNet-101 | ~44.5M     |
| ResNet-152 | ~60.2M     |

---

## 🔹 Strengths

* Trains **100+ layer networks**
* Excellent gradient flow
* General-purpose backbone:

  * Object Detection (Faster R-CNN)
  * Segmentation (Mask R-CNN)
  * Video, Medical Imaging
  * Even NLP adaptations

---

## 🔹 Weaknesses

* High computation & memory cost
* Identity shortcut not always optimal
* Outperformed by:

  * DenseNet
  * EfficientNet
  * ConvNeXt (modern CNNs)

---

## 🔹 Final Summary (Interview-Ready)

> **ResNet introduced residual learning using identity skip connections, allowing networks to learn residual functions instead of full mappings. This solved the degradation and vanishing gradient problems, enabling extremely deep neural networks and becoming the foundation of modern computer vision architectures.**

---

If you want next:

* 🔥 **ResNet vs DenseNet vs EfficientNet**
* 🔥 **Why ResNet still beats ViT in low-data regimes**
* 🔥 **PyTorch implementation from scratch**
* 🔥 **Interview Q&A (FAANG-level)**
