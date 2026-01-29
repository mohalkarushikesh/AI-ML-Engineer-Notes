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

## 🏗️ Architecture Breakdown
| Layer Type | Details | Innovations |
|------------|---------|-------------|
| **Conv1** | 96 kernels of size 11×11×3, stride 4 | ReLU activation, Local Response Normalization (LRN) |
| **MaxPool1** | 3×3 kernel, stride 2 | Reduces spatial dimensions |
| **Conv2** | 256 kernels of size 5×5 | Split across GPUs for efficiency |
| **MaxPool2** | 3×3 kernel, stride 2 | Further downsampling |
| **Conv3** | 384 kernels of size 3×3 | Deeper feature extraction |
| **Conv4** | 384 kernels of size 3×3 | More complex features |
| **Conv5** | 256 kernels of size 3×3 | Final convolutional layer |
| **MaxPool3** | 3×3 kernel, stride 2 | Final pooling |
| **FC1** | 4096 neurons | Dropout (p=0.5) applied |
| **FC2** | 4096 neurons | Dropout again |
| **FC3** | 1000 neurons | Softmax output |

<img width='800' height='600' src="https://github.com/user-attachments/assets/964c66a2-bde6-4525-a86d-2f3c6b982cdc" />   

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
