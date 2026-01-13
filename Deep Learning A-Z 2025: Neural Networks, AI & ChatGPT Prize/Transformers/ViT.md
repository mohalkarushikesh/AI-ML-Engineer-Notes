Here are **clean, exam + interview + implementation–ready notes on Vision Transformer (ViT)**, structured for fast revision 👇

---

# Vision Transformer (ViT) – Complete Notes

## 1. What is ViT?

**Vision Transformer (ViT)** applies the **Transformer architecture (used in NLP)** directly to images by treating an image as a **sequence of patches**.

📌 Instead of convolutions (CNNs), ViT uses:

* **Self-attention**
* **Global context modeling**
* **Patch embeddings**

---

## 2. Why ViT over CNN?

| CNN                                    | ViT                           |
| -------------------------------------- | ----------------------------- |
| Local receptive field                  | Global attention from start   |
| Hard-coded inductive bias              | Learns relationships freely   |
| Efficient for small data               | Needs large-scale pretraining |
| Struggles with long-range dependencies | Excels at global dependencies |

👉 ViT shines when **large datasets (ImageNet-21k, JFT-300M)** are available.

---

## 3. ViT Architecture (High Level)

```
Image → Patches → Linear Projection → Transformer Encoder → MLP Head
```

---

## 4. Step-by-Step ViT Pipeline

### 4.1 Image to Patches

* Input image: `H × W × C`
* Patch size: `P × P`
* Number of patches:

```
N = (H × W) / P²
```

Each patch is **flattened** into a vector.

---

### 4.2 Patch Embedding

* Flattened patch → Linear projection
* Output dimension: `D`

📌 Similar to **word embeddings** in NLP.

---

### 4.3 Class Token ([CLS])

* A **learnable token** added to patch sequence
* Final classification is based on this token

```
[CLS] + Patch₁ + Patch₂ + ... + Patchₙ
```

---

### 4.4 Positional Encoding

Transformers lack order awareness → add **positional embeddings**:

```
Z₀ = [x_cls ; x₁ ; x₂ ; ... ; xₙ] + E_pos
```

* Learned positional embeddings (not sinusoidal)

---

## 5. Transformer Encoder (Core of ViT)

Each encoder block has:

### 5.1 Multi-Head Self Attention (MHSA)

```
Attention(Q,K,V) = softmax(QKᵀ / √d) V
```

* Captures **global relationships**
* Computational complexity: **O(N²)**

---

### 5.2 Feed Forward Network (MLP)

```
MLP(x) = GELU(xW₁ + b₁)W₂ + b₂
```

---

### 5.3 Residual + LayerNorm

```
x = x + MHSA(LN(x))
x = x + MLP(LN(x))
```

📌 Uses **Pre-LayerNorm** (better stability)

---

## 6. Classification Head

* Take **[CLS] token output**
* Pass through MLP head
* Softmax → class probabilities

---

## 7. ViT Variants

| Model    | Patch Size | Hidden Dim | Heads |
| -------- | ---------- | ---------- | ----- |
| ViT-B/16 | 16×16      | 768        | 12    |
| ViT-L/16 | 16×16      | 1024       | 16    |
| ViT-H/14 | 14×14      | 1280       | 16    |

---

## 8. Training Strategy

### 8.1 Pretraining

* Large datasets required
* Self-supervised or supervised

### 8.2 Fine-tuning

* On smaller downstream datasets
* Often freeze early layers

---

## 9. Limitations of ViT

❌ Needs **huge data**
❌ Computationally expensive (attention is O(N²))
❌ Poor inductive bias for images

---

## 10. Improvements over ViT

| Model            | Key Idea                              |
| ---------------- | ------------------------------------- |
| DeiT             | Data-efficient training               |
| Swin Transformer | Window-based attention                |
| BEiT             | Self-supervised masked image modeling |
| ConvNeXt         | CNN redesigned using ViT insights     |

---

## 11. ViT vs CNN (Interview Answer)

> **CNNs rely on locality and translation invariance, while ViTs rely on self-attention to capture global context. ViTs scale better with data but require large datasets and compute.**

---

## 12. When to Use ViT?

✅ Large datasets
✅ Need global context
✅ Pretrained models available

❌ Small datasets
❌ Low compute environments

---

## 13. Key Formulas to Remember

* Number of patches:

  ```
  N = (H × W) / P²
  ```
* Attention:

  ```
  softmax(QKᵀ / √d)
  ```

---

## 14. One-Line Summary

> **ViT treats an image as a sequence of patches and uses Transformer encoders to learn global visual representations
