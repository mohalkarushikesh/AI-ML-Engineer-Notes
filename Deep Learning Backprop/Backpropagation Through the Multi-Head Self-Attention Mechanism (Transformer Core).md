### Topic: **Backpropagation Through the Multi-Head Self-Attention Mechanism (Transformer Core)**

This is the **mathematical heart** of **GPT, BERT, Llama, Grok**, etc.  
We’ll walk **backward** through **one forward pass of Multi-Head Self-Attention**, showing **exact gradient flow** with **full LaTeX + code-level intuition**.

---

## Forward Pass Recap (One Head)

Given input embeddings: \( X \in \mathbb{R}^{T \times d} \) (T = sequence length, d = model dim)

### Step 1: Linear Projections
\[
Q = X W_Q, \quad K = X W_K, \quad V = X W_V \quad \in \mathbb{R}^{T \times d_k}
\]
\( W_Q, W_K, W_V \in \mathbb{R}^{d \times d_k} \), \( d_k = d / h \)

### Step 2: Scaled Dot-Product Attention
\[
\text{Attn}(Q, K, V) = \underbrace{\text{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right)}_{\text{Attention Weights } A} V
\]

Let:
- \( S = Q K^T \in \mathbb{R}^{T \times T} \) → raw scores
- \( A = \text{softmax}(S / \sqrt{d_k}) \) → probabilities
- \( Z = A V \in \mathbb{R}^{T \times d_k} \) → output

---

## Goal of Backpropagation
Given loss \( L \), compute:
\[
\frac{\partial L}{\partial X}, \quad \frac{\partial L}{\partial W_Q}, \frac{\partial L}{\partial W_K}, \frac{\partial L}{\partial W_V}
\]

Let upstream gradient: \( \frac{\partial L}{\partial Z} = G_Z \in \mathbb{R}^{T \times d_k} \)

---

## Backward Pass (Step-by-Step)

---

### **1. Gradient w.r.t. \( V \)**: \( \frac{\partial L}{\partial V} \)

\[
Z = A V \quad \Rightarrow \quad \frac{\partial L}{\partial V} = A^T G_Z
\]

> Shape: \( (T \times T)^T \cdot (T \times d_k) = T \times d_k \)

---

### **2. Gradient w.r.t. \( A \)**: \( \frac{\partial L}{\partial A} \)

\[
\frac{\partial L}{\partial A} = G_Z V^T
\]

> Shape: \( (T \times d_k) \cdot (d_k \times T) = T \times T \)

---

### **3. Gradient w.r.t. \( S \)** (pre-softmax): **Softmax Jacobian**

\[
A_i = \text{softmax}(S_i), \quad \frac{\partial A_{ij}}{\partial S_{ik}} = A_{ij} (\delta_{jk} - A_{ik})
\]

So:
\[
\frac{\partial L}{\partial S} = A \odot \left( \frac{\partial L}{\partial A} - \sum_j \frac{\partial L}{\partial A}_{:j} A_{:j} \right)
\]

**Efficient form**:
\[
\frac{\partial L}{\partial S} = \frac{1}{\sqrt{d_k}} \cdot \left( A \odot (G_Z V^T) - A \odot (A \cdot (G_Z V^T)) \right)
\]

Or in code:
```python
dA = G_Z @ V.T
grad_S = (A * (dA - (A * dA).sum(dim=-1, keepdim=True))) / sqrt(d_k)
```

---

### **4. Gradient w.r.t. \( K \)** and \( Q \)

\[
S = \frac{1}{\sqrt{d_k}} Q K^T
\]

\[
\frac{\partial L}{\partial K} = \frac{1}{\sqrt{d_k}} Q^T \cdot \frac{\partial L}{\partial S}
\]
\[
\frac{\partial L}{\partial Q} = \frac{1}{\sqrt{d_k}} \frac{\partial L}{\partial S} \cdot K
\]

> Both: \( (T \times d_k) \)

---

### **5. Gradient w.r.t. \( X \)** (from Q, K, V)

\[
\frac{\partial L}{\partial X} = 
\underbrace{\frac{\partial L}{\partial Q} W_Q^T}_{\text{from } Q} + 
\underbrace{\frac{\partial L}{\partial K} W_K^T}_{\text{from } K} + 
\underbrace{\frac{\partial L}{\partial V} W_V^T}_{\text{from } V}
\]

---

### **6. Gradient w.r.t. Weights**

\[
\frac{\partial L}{\partial W_Q} = X^T \cdot \frac{\partial L}{\partial Q}, \quad
\frac{\partial L}{\partial W_K} = X^T \cdot \frac{\partial L}{\partial K}, \quad
\frac{\partial L}{\partial W_V} = X^T \cdot \frac{\partial L}{\partial V}
\]

---

## Multi-Head? Just Concat + Linear

After \( h \) heads: \( Z_{\text{multi}} = \text{Concat}(Z_1, \dots, Z_h) W_O \)

Backward:
- Split \( G_{Z_{\text{multi}}} W_O^T \) → per head
- Run above per head
- Sum \( \frac{\partial L}{\partial X} \) across heads

---

## Full Gradient Flow Summary (One Head)

\[
\boxed{
\begin{aligned}
G_Z &\xrightarrow{\text{V}} A^T G_Z && \to \frac{\partial L}{\partial V} \\
&\xrightarrow{\text{A}} G_Z V^T && \to \text{softmax}^{-1} \to \frac{\partial L}{\partial S} \\
&\xrightarrow{\text{S}} \frac{1}{\sqrt{d_k}} (Q^T \frac{\partial L}{\partial S}, \frac{\partial L}{\partial S} K) && \to \frac{\partial L}{\partial K}, \frac{\partial L}{\partial Q} \\
&\xrightarrow{\text{X}} \sum (\frac{\partial L}{\partial Q} W_Q^T + \frac{\partial L}{\partial K} W_K^T + \frac{\partial L}{\partial V} W_V^T)
\end{aligned}
}
\]

---

## PyTorch-Style Pseudocode (Autograd Hides This)

```python
class AttentionHead(nn.Module):
    def forward(self, X):
        Q = X @ self.W_Q
        K = X @ self.W_K
        V = X @ self.W_V
        S = Q @ K.T / sqrt(d_k)
        A = torch.softmax(S, dim=-1)
        Z = A @ V
        return Z

    # Backward: PyTorch computes all above automatically!
```

But **you now know the math**.

---

## Why This Matters

| Issue | Attention Solves |
|------|------------------|
| **Sequential RNNs** | O(T) dependency → slow |
| **Fixed context** | Attention = **dynamic memory** |
| **Vanishing grads** | Direct \( X \to Z \) paths |

---

## Challenge Question

> If **causal masking** is added (lower triangle zeroed before softmax), how does \( \frac{\partial L}{\partial S} \) change?

**Answer**:  
Mask \( M \): \( S = S + M \) (M = −∞ where future)  
→ \( A \) unchanged for valid positions  
→ But **gradient still flows only within allowed positions** due to softmax zeroing.

---

## Next Complex Topics (Pick One)

| Topic | Why It's Hard |
|------|---------------|
| **FlashAttention-2 Gradient Flow** | Fused kernel, tiling, no materialization |
| **KV Cache + Rotary Embeddings Backprop** | Positional, incremental |
| **Sparse Attention (Reformer, BigBird)** | Locality + hashing |
| **Gated Linear Units (GLU) in Feedforward** | Non-linearity + gating gradients |

---

**Say the name** → I’ll go **full math + code + kernel-level**.
