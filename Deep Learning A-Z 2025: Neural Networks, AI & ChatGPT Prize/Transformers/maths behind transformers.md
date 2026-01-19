The mathematics behind the **Transformer model** is built on *linear algebra and probability*, especially the **attention mechanism**. At its core, Transformers compute relationships between tokens using **queries, keys, and values**, apply **scaled dot-product attention**, and combine this with **positional encoding, normalization, and feed-forward layers** to model sequences efficiently.

---

# 🧮 Detailed Mathematics of Transformers

## 1. Input Representation
- Each token is mapped to a vector embedding:  
  $$x_i \in \mathbb{R}^d$$  
- Positional encoding adds sequence order:  
  $$PE_{(pos,2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos,2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$  
- Final input:  
  $$z_i = x_i + PE_i$$  

---

## 2. Self-Attention Mechanism
- Define **Query (Q), Key (K), Value (V)** matrices:  
  $$Q = ZW^Q, \quad K = ZW^K, \quad V = ZW^V$$  
  where $W^Q, W^K, W^V$ are learned weight matrices.  

- **Attention scores:**  
  $$\text{Scores} = QK^T$$  

- **Scaled dot-product attention:**  
  $$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$  

- This computes weighted averages of values $V$, where weights depend on similarity between queries and keys.

---

## 3. Multi-Head Attention
- Instead of one attention, use $h$ heads:  
  $$\text{MHA}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$  
- Each head:  
  $$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$  

👉 This allows the model to capture different types of relationships simultaneously.

---

## 4. Feed-Forward Network
- After attention, each token passes through a position-wise feed-forward network:  
  $$FFN(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$$  

---

## 5. Normalization & Residuals
- **Layer normalization:**  
  $$\text{LayerNorm}(x) = \frac{x - \mu}{\sigma} \cdot \gamma + \beta$$  
- **Residual connections:** Add input back to output for stability:  
  $$y = x + \text{SubLayer}(x)$$  

---

## 6. Encoder-Decoder Structure
- **Encoder:** Stacks of self-attention + feed-forward layers.  
- **Decoder:** Similar, but includes *cross-attention* with encoder outputs.  
- Cross-attention formula:  
  $$\text{Attention}(Q_{dec}, K_{enc}, V_{enc})$$  

---

## 📊 Summary Table

| Component             | Formula / Operation |
|-----------------------|---------------------|
| Positional Encoding   | $PE_{(pos,2i)} = \sin(\frac{pos}{10000^{2i/d}})$ |
| Attention Scores      | $QK^T$ |
| Scaled Attention      | $\text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$ |
| Multi-Head Attention  | $\text{Concat}(\text{head}_i)W^O$ |
| Feed-Forward          | $\text{ReLU}(xW_1+b_1)W_2+b_2$ |
| Normalization         | $\frac{x-\mu}{\sigma}\cdot\gamma+\beta$ |

---

## ⚡ Key Insights
- **Transformers replace recurrence with attention**, enabling parallelization.  
- **Mathematics is dominated by matrix multiplications and softmax weighting.**  
- **Quadratic complexity $(O(n^2)$ )** in sequence length due to pairwise attention.  

---

**In short:** Transformers are mathematically defined by 
- **embedding + positional encoding → attention (QKV) → multi-head → feed-forward → normalization → residuals**, stacked in encoder-decoder layers.  

Sources: [ArXiv: Mathematical Explanation of Transformers](https://arxiv.org/pdf/2510.03989), [Transformer Deep Dive Tutorial](https://www.zanganehai.com/tutorials/transformers/)

---

## 🧩 Example: Sentence = “I love AI”

### Step 1: Embedding + Position
- Each word → vector (say 3 numbers for simplicity).  
- Add position info so the model knows order.

---

### Step 2: Queries, Keys, Values
- For each word, create:
  - **Query (Q):** “What am I looking for?”  
  - **Key (K):** “What do I offer?”  
  - **Value (V):** “My actual content.”  

Example (tiny numbers):  
- “I”: Q=[1], K=[1], V=[2]  
- “love”: Q=[2], K=[2], V=[3]  
- “AI”: Q=[3], K=[3], V=[4]  

---

### Step 3: Attention
- Compute similarity: Q × K.  
- For “love”:  
  - Compare with “I” → 2×1 = 2  
  - Compare with “love” → 2×2 = 4  
  - Compare with “AI” → 2×3 = 6  

- Scale + softmax → weights (like probabilities).  
  Example weights: [0.1, 0.3, 0.6].

---

### Step 4: Weighted Sum
- Multiply weights × Values:  
  $$0.1 \cdot 2 + 0.3 \cdot 3 + 0.6 \cdot 4 = 3.7$$  

So “love” now becomes a new vector that blends info from “I” and “AI”.

---

### Step 5: Multi-Head + Feed Forward
- Do this with multiple heads (different perspectives).  
- Pass through a small neural net → final representation.

---

## 🎯 In Short
Transformers = **Each word looks at all other words, decides how much to pay attention, and updates itself accordingly.**  
That’s why “love” understands it’s between “I” and “AI”.

---
