**VGG (Visual Geometry Group network)** — one of the landmark CNN architectures after LeNet and AlexNet:

---

## 📘 VGG Architecture (2014, Oxford VGG team)

### 🔹 Key Features
- Introduced by **Karen Simonyan & Andrew Zisserman** in the paper *“Very Deep Convolutional Networks for Large-Scale Image Recognition”*.  
- Famous for its **simplicity**: only stacked convolutional layers with small filters (3×3).  
- Depth: **16 or 19 layers** (VGG16, VGG19).  
- Achieved **top results in ImageNet 2014**.  

---

### 🔹 Structure
1. **Input**  
   - Image size: $(224 \times 224 \times 3\)$ (RGB).  

2. **Convolutional Blocks**  
   - Multiple **3×3 convolution filters** stacked.  
   - Each block followed by **2×2 max pooling**.  
   - Depth increases gradually (64 → 128 → 256 → 512 filters).  

3. **Fully Connected Layers**  
   - Three dense layers: two with 4096 units, one with 1000 units (for ImageNet classes).  

4. **Output Layer**  
   - Softmax classifier for 1000 categories.  

<img width='1200' height= '450' src="https://github.com/user-attachments/assets/8f1527a2-75e1-435f-9785-b7daaef9ce52" />

---

### 🔹 Maths Behind VGG
- **Convolution:**
```math
  y_{i,j}^{(k)} = \sum_{m=0}^{2} \sum_{n=0}^{2} x_{i+m, j+n} \cdot w_{m,n}^{(k)} + b^{(k)}
```  
  (3×3 filters, stride 1).  

- **Pooling:**  
  Max pooling with 2×2 window, stride 2.  

- **Fully Connected:**  
  Standard dense layer:  
```math
  y = W \cdot x + b
```

---

### 🔹 Parameter Count
- **VGG16:** ~138 million parameters.  
- **VGG19:** ~143 million parameters.  
- Much larger than LeNet (~61k) and AlexNet (~60M).  

---

### 🔹 Strengths
- Simple, uniform design (only 3×3 conv filters).  
- Very deep compared to predecessors.  
- Became a **benchmark backbone** for transfer learning.  

### 🔹 Weaknesses
- Extremely **large parameter count** → heavy memory and compute requirements.  
- Training is slow compared to modern architectures.  

---

✅ **In short:** VGG proved that **depth matters** in CNNs. By stacking small filters, it achieved high accuracy, but at the cost of massive parameter size.

---
