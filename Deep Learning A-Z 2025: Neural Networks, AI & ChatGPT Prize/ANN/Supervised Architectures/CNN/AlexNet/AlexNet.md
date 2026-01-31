# 📘 AlexNet Notes

**AlexNet** is a landmark deep learning model introduced in 2012 that revolutionized computer vision by winning the ImageNet competition with a massive performance leap. It consists of 8 layers (5 convolutional + 3 fully connected), uses ReLU activations, dropout, and data augmentation to combat overfitting, and processes **227×227 RGB images** into 1000-class predictions.

---

## 🏆 Overview
- **Introduced by:** Krizhevsky, Sutskever, and Hinton (2012)  
- **Competition Impact:** Won ILSVRC 2012 with **top-5 error rate of 15.3%** (runner-up was 26.2%)  
- **Input:** 227×227×3 RGB images  
- **Output:** Probabilities over 1000 ImageNet classes  
- **Parameters:** ~60–62 million trainable parameters  

---

<img width='800' height='600' src="https://github.com/user-attachments/assets/964c66a2-bde6-4525-a86d-2f3c6b982cdc" />   

## 🏗️ Architecture Breakdown

| **Layer Type** | **Mathematical Details** | **Innovations** |
|----------------|---------------------------|-----------------|
| **Conv1** | $(11 \times 11 \times 3)$ kernels, stride 4 → Output size: $55 \times 55 \times 96$. Each neuron computes: $y = \sum w_{ij}x_{ij} + b$ | Introduced **ReLU** activation (faster convergence), Local Response Normalization (LRN) |
| **MaxPool1** | $3 \times 3$ kernel, stride 2 → Output size: $27 \times 27 \times 96$. Operation: $y = \max(x_{ij})$ | Reduced spatial dimensions, improved invariance |
| **Conv2** | $5 \times 5$ kernels, 256 filters → Output size: $27 \times 27 \times 256$. Each neuron: $y = \sum w_{ij}x_{ij} + b$ | Split across **two GPUs** for efficiency |
| **MaxPool2** | $3 \times 3$ kernel, stride 2 → Output size: $13 \times 13 \times 256$ | Downsampling, reduced computation |
| **Conv3** | $3 \times 3$ kernels, 384 filters → Output size: $13 \times 13 \times 384$ | Deeper feature extraction |
| **Conv4** | $3 \times 3$ kernels, 384 filters → Output size: $13 \times 13 \times 384$ | Captures more complex features |
| **Conv5** | $3 \times 3$ kernels, 256 filters → Output size: $13 \times 13 \times 256$ | Final convolutional layer |
| **MaxPool3** | $3 \times 3$ kernel, stride 2 → Output size: $6 \times 6 \times 256$ | Final pooling before dense layers |
| **FC1** | Flattened input: $6 \times 6 \times 256 = 9216$. Fully connected to 4096 neurons: $y = W \cdot x + b$ | Dropout ($p=0.5$) to reduce overfitting |
| **FC2** | 4096 neurons fully connected: $y = W \cdot x + b$ | Dropout again |
| **FC3** | 1000 neurons (ImageNet classes). Softmax: $P(y=i \mid x) = \frac{e^{z_i}}{\sum_j e^{z_j}}$ | Final classification layer |


---

### 📘 AlexNet Layer-by-Layer Breakdown

#### **1. Conv1**
- **Math:** $( (11 \times 11 \times 3) \)$ kernels, stride 4 → output size $(55 \times 55 \times 96\)$  
```math
  y_{i,j}^{(k)} = \sum_{m,n,c} w_{m,n,c}^{(k)} \cdot x_{i+m, j+n, c} + b^{(k)}
```
- **Details:** 96 filters, large receptive field.  
- **Innovations:** ReLU activation, Local Response Normalization (LRN).

---

#### **2. MaxPool1**
- **Math:** $(3 \times 3\)$ kernel, stride 2 → output size $(27 \times 27 \times 96\)$ 
```math
  y_{i,j} = \max_{m,n} x_{i+m, j+n}
```
- **Details:** Downsampling.  
- **Innovations:** Early spatial reduction.

---

#### **3. Conv2**
- **Math:** $(5 \times 5\)$ kernels, 256 filters → output size $(27 \times 27 \times 256\)$  
- **Details:** Split across GPUs for efficiency.  
- **Innovations:** Parallelization across GPUs.

---

#### **4. MaxPool2**
- **Math:** $(3 \times 3\)$ kernel, stride 2 → output size $(13 \times 13 \times 256\)$  
- **Details:** Further downsampling.  
- **Innovations:** Reduces computation.

---

#### **5. Conv3**
- **Math:** $(3 \times 3\)$ kernels, 384 filters → output size $(13 \times 13 \times 384\)$  
- **Details:** Deeper feature extraction.  
- **Innovations:** Increased depth.

---

#### **6. Conv4**
- **Math:** $(3 \times 3\)$ kernels, 384 filters → output size $(13 \times 13 \times 384\)$  
- **Details:** Builds on Conv3 features.  
- **Innovations:** More complex features.

---

#### **7. Conv5**
- **Math:** $(3 \times 3\)$ kernels, 256 filters → output size $(13 \times 13 \times 256\)$  
- **Details:** Final convolutional layer.  
- **Innovations:** Prepares for dense layers.

---

#### **8. MaxPool3**
- **Math:** $(3 \times 3\)$ kernel, stride 2 → output size $(6 \times 6 \times 256\)$  
- **Details:** Final pooling.  
- **Innovations:** Transition to fully connected layers.

---

#### **9. FC1**
- **Math:** Flattened input: $(6 \times 6 \times 256 = 9216\)$ Fully connected to 4096 neurons.  
```math
  y = W \cdot x + b
```
- **Details:** Dense layer with 4096 units.  
- **Innovations:** Dropout (p=0.5).

---

#### **10. FC2**
- **Math:** 4096 neurons fully connected.  
- **Details:** Another dense layer.  
- **Innovations:** Dropout again.

---

#### **11. FC3 (Output)**
- **Math:** 1000 neurons (ImageNet classes). Softmax:  
```math
  P(y=i|x) = \frac{e^{z_i}}{\sum_{j=1}^{1000} e^{z_j}}
```
- **Details:** Final classification layer.  
- **Innovations:** Large-scale classification.

---

✅ **Summary:** AlexNet’s math is convolution, pooling, dense layers, and softmax. Its innovations — **ReLU, LRN, dropout, GPU parallelization** — made deep CNNs practical and trainable at scale.

---


## ⚡ Key Innovations
- **ReLU Activation:** Faster training vs. sigmoid/tanh, avoids vanishing gradients  
- **Dropout:** Prevents overfitting by randomly deactivating neurons  
- **Data Augmentation:** Translations, reflections, patch extractions expanded training set  
- **GPU Utilization:** Training split across two GTX 580 GPUs  

---

## 🌍 Importance & Legacy
- **Breakthrough:** Proved deep CNNs outperform traditional vision methods  
- **Foundation:** Inspired VGG, ResNet, EfficientNet  
- **Impact:** Sparked widespread adoption of deep learning in academia & industry  

---

## 📌 Quick Notes for Study
- **Layers:** 8 (5 conv + 3 FC)  
- **Parameters:** ~60–62M  
- **Activations:** ReLU  
- **Regularization:** Dropout + Data Augmentation  
- **Training:** Two GPUs, batch size 128  
- **Impact:** Reduced ImageNet error rate by ~40%  

---
