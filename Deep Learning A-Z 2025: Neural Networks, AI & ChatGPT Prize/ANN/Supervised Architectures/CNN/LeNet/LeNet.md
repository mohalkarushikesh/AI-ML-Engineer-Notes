## 📘 LeNet Architecture (1998, Yann LeCun)

### 🔹 Key Features
- Designed for **handwritten digit recognition** (MNIST dataset).  
- One of the first successful CNNs, pioneering deep learning in computer vision.  
- Uses convolution + pooling layers followed by fully connected layers.  

---

### 🔹 Structure
1. **Input Layer**  
   - Grayscale image (e.g., 32×32 pixels).  

2. **Convolutional Layer (C1)**  
   - 6 filters of size 5×5.  
   - Produces 6 feature maps.  

3. **Pooling Layer (S2)**  
   - Average pooling (subsampling).  
   - Reduces spatial dimensions.  

4. **Convolutional Layer (C3)**  
   - 16 filters of size 5×5.  
   - Produces 16 feature maps.  

5. **Pooling Layer (S4)**  
   - Average pooling again.  

6. **Fully Connected Layers (C5, F6)**  
   - Dense layers for classification.  

7. **Output Layer**  
   - Softmax classifier (e.g., 10 classes for digits 0–9).  

---

### 🔹 Characteristics
- **Activation Function:** Sigmoid or tanh (ReLU wasn’t popular yet).  
- **Pooling:** Average pooling (modern CNNs use max pooling).  
- **Parameters:** Much fewer compared to modern CNNs.  
- **Strength:** Showed CNNs could outperform traditional methods in vision tasks.  

---

### 🔹 Applications
- Handwritten digit recognition (MNIST).  
- Early document recognition systems.  
- Foundation for modern CNNs like AlexNet, VGG, ResNet.  

---

✅ **In short:** LeNet is the pioneering CNN that introduced convolution + pooling + fully connected layers for image recognition, laying the groundwork for modern deep learning in computer vision.

---

<img width="952" height="480" alt="lenet-min" src="https://github.com/user-attachments/assets/917a746d-ed2f-4a7c-be0d-af3d932e2804" />

---

✅ **This flow shows:**  
- Input image → convolution → pooling → deeper convolution → pooling → fully connected layers → output.  
- It’s the classic CNN pipeline that inspired modern architectures like AlexNet, VGG, and ResNet.  



Here’s the **mathematical backbone of LeNet** — the operations that happen layer by layer:

---

## 📘 Maths Behind LeNet

### 1. **Convolution Layers (C1, C3)**
- Operation:  
```math
  y_{i,j}^{(k)} = \sum_{m=0}^{f-1} \sum_{n=0}^{f-1} x_{i+m, j+n} \cdot w_{m,n}^{(k)} + b^{(k)}
```
  - $(x\)$ : input pixel values  
  - $(w\)$ : filter weights  
  - $(b\)$ : bias term  
  - $(f\)$ : filter size (5×5 in LeNet)  
- Produces **feature maps** by sliding filters over the input image.

---

### 2. **Pooling Layers (S2, S4)**
- LeNet used **average pooling** (subsampling).  
- Operation:
```math
  y_{i,j} = \frac{1}{p^2} \sum_{m=0}^{p-1} \sum_{n=0}^{p-1} x_{i \cdot p + m, j \cdot p + n}
```
  - $(p\)$ : pooling window size (2×2 in LeNet).  
- Reduces spatial dimensions while retaining important features.

---

### 3. **Fully Connected Layers (C5, F6)**
- Standard dense layer:  
```math
  y = W \cdot x + b
```
  - $(W\)$ : weight matrix  
  - $(x\)$ : flattened input vector  
  - $(b\)$ : bias  
- Each neuron connects to **all inputs** from the previous layer.

---

### 4. **Activation Functions**
- LeNet used **sigmoid** or **tanh** (not ReLU).  
- Sigmoid:  
```math
  \sigma(z) = \frac{1}{1 + e^{-z}}
```
- Tanh:  
```math
  \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}
```

---

### 5. **Output Layer (Softmax)**
- For classification into 10 digits (0–9).  
- Operation:  
```math
  P(y=i|x) = \frac{e^{z_i}}{\sum_{j=1}^{10} e^{z_j}}
```
  - Converts raw scores into probabilities.

---

## 🔹 Summary
- **Convolution:** weighted sums over local regions.  
- **Pooling:** dimension reduction by averaging.  
- **Fully Connected:** matrix multiplication.  
- **Activation:** nonlinear transformation (sigmoid/tanh).  
- **Softmax:** probability distribution over classes.  

---

Perfect — let’s break down the **parameter count layer by layer in LeNet-5** (the most cited version).  

---

## 📘 Parameter Count in LeNet-5

### 1. **Conv Layer C1**
- Input: $(32 \times 32 \times 1\)$ (grayscale image)  
- Filters: 6 filters, each $(5 \times 5\)$  
- Parameters per filter: $(5 \times 5 \times 1 + 1 = 26\)$ (weights + bias)  
- Total: $(26 \times 6 = 156\)$  

---

### 2. **Conv Layer C3**
- Input: $(14 \times 14 \times 6\)$
- Filters: 16 filters, each $(5 \times 5\)$  
- Parameters per filter: $(5 \times 5 \times 6 + 1 = 151\)$
- Total: $(151 \times 16 = 2,416\)$  

---

### 3. **Fully Connected Layer C5**
- Input: $(5 \times 5 \times 16 = 400\)$  
- Neurons: 120  
- Parameters: $(400 \times 120 + 120 = 48,120\)$  

---

### 4. **Fully Connected Layer F6**
- Input: 120  
- Neurons: 84  
- Parameters: $(120 \times 84 + 84 = 10,164\)$  

---

### 5. **Output Layer**
- Input: 84  
- Neurons: 10 (digits 0–9)  
- Parameters: $(84 \times 10 + 10 = 850\)$  

---

## 🔹 Total Parameters
```math
156 + 2,416 + 48,120 + 10,164 + 850 = 61,706
```

So **LeNet-5 has ~61,700 trainable parameters** — tiny compared to modern CNNs (AlexNet has ~60 million, ResNet-50 has ~25 million).  

---

✅ **Key Insight:**  
LeNet was small enough to run on 1990s hardware, yet powerful enough to prove CNNs could outperform traditional vision methods.  
