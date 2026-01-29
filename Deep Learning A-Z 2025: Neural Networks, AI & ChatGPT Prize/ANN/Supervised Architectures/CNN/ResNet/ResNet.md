## 📘 ResNet Architecture (2015, Microsoft Research)

### 🔹 Key Idea
- Introduced in the paper *“Deep Residual Learning for Image Recognition”*.  
- Solved the **vanishing gradient problem** in very deep networks.  
- Used **skip connections (residual connections)** to allow gradients to flow directly through layers.  

---

### 🔹 Structure
1. **Input**  
   - ImageNet input size: $(224 \times 224 \times 3\)$  

2. **Convolution + Pooling**  
   - Initial 7×7 convolution, stride 2.  
   - 3×3 max pooling.  

3. **Residual Blocks**  
   - Each block has:  
     - Convolution layers (3×3 filters).  
     - **Skip connection:**  
     
     $y = F(x, W) + x$
     
     where $(F(x, W)\)$ is the transformation (convolutions), and $(x\)$ is the identity shortcut.  

4. **Stacked Residual Layers**  
   - Depth varies: ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152.  
   - Bottleneck design (ResNet-50 and deeper): 1×1 → 3×3 → 1×1 convolutions.  

5. **Fully Connected + Softmax**  
   - Dense layer for classification (1000 ImageNet classes).  

<img width="688" height="267" alt="The-structure-of-the-ResNet34-CNN-Network-The-input-of-the-network-is-the-preprocessed" src="https://github.com/user-attachments/assets/fe65dadf-4d8a-43b4-ad90-f1c3c9dc4712" />

---

### 🔹 Maths Behind Residual Connections
- Standard layer:  
```math
  y = F(x, W)
```  
- Residual layer:  
```math
  y = F(x, W) + x
```  
- This allows the network to **learn residuals** instead of full transformations, making training easier for very deep networks.

---

### 🔹 Parameter Counts
- **ResNet-18:** ~11 million parameters.  
- **ResNet-34:** ~21 million.  
- **ResNet-50:** ~25 million.  
- **ResNet-101:** ~44 million.  
- **ResNet-152:** ~60 million.  

---

### 🔹 Strengths
- Enabled training of networks with **hundreds of layers**.  
- Skip connections prevent vanishing gradients.  
- Became the backbone for many tasks: detection, segmentation, NLP.  

### 🔹 Weaknesses
- Still computationally heavy compared to newer architectures (e.g., EfficientNet).  
- Large memory footprint.  

---

✅ **In short:** ResNet revolutionized deep learning by introducing residual connections, allowing networks to go much deeper without losing trainability.

---
