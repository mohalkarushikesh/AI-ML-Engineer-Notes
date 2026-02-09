 The mathematics behind the **Mamba framework** is built on *Selective State Space Models (SSSMs)*. It combines the continuous-time dynamics of **state space models** with **input-dependent gating**, allowing linear-time sequence modeling while retaining content-awareness — something traditional SSMs struggled with.  

---

# 🧮 Mathematical Foundations of Mamba

## 1. **State Space Models (SSMs)**
- General form of a continuous-time linear dynamical system:  

$$\frac{d}{dt}x(t) = A x(t) + B u(t), \quad y(t) = C x(t) + D u(t)$$  

- Where:  
  - $x(t)$: hidden state  
  - $u(t)$: input sequence  
  - $y(t)$: output sequence  
  - $A, B, C, D$: system matrices  

- Discretization (for sequences):  

$$x_{t+1} = A_d x_t + B_d u_t, \quad y_t = C_d x_t + D u_t$$  

---

## 2. **Selective State Space Models (SSSMs)**
- Mamba introduces **selectivity**: parameters $A, B, C$ are *input-dependent*.  
- Instead of fixed matrices, they vary with the token:  

$$x_{t+1} = A(u_t) x_t + B(u_t) u_t$$  

$$y_t = C(u_t) x_t$$  

- This allows the model to **propagate or forget information dynamically**, depending on the current input.  

---

## 3. **Linear-Time Complexity**
- Traditional Transformers: $(O(n^2)\)$ due to pairwise attention.  
- Mamba: Uses **convolutions with structured kernels** derived from the state space equations.  
- Efficient GPU implementation (similar to FlashAttention) ensures:

$$O(n) \quad \text{time complexity for sequence length } n$$

---

## 4. **Dynamic Gating**
- A gating function $(g(u_t)\)$ decides whether to update or skip state transitions.  
- Mathematically:

$$x_{t+1} = g(u_t) \cdot (A x_t + B u_t) + (1 - g(u_t)) \cdot x_t$$
  
- This mechanism gives Mamba **content-based reasoning** — something earlier SSMs lacked.

---

## 5. **Training Objective**
- Like Transformers, Mamba is trained end-to-end with gradient descent.  
- Loss functions: typically cross-entropy for language modeling.  
- Optimization benefits from **linear recurrence kernels** and GPU-friendly algorithms.

---

# 📊 Summary Table

| Component              | Transformer (Attention) | Mamba (SSSM) |
|------------------------|--------------------------|--------------|
| Core math              | Dot-product attention   | State-space recurrence |
| Complexity             | $(O(n^2)\)$             | $(O(n)\)$ |
| Parameters             | Fixed weights           | Input-dependent matrices |
| Memory                 | KV-cache heavy          | Efficient, linear |
| Selectivity            | Global attention        | Dynamic gating |

---

**Takeaway:**  
Mamba’s mathematics is essentially **state space modeling with selective, input-dependent updates**, enabling linear-time sequence processing while retaining the ability to reason about content — a major step toward replacing Transformers in long-context tasks.  

---

# 🧮 Toy Example of Mamba Sequence Processing


## 1. Setup
We have a short input sequence:  

$$u = [2, 5, 3]$$
  
Think of these as three tokens (numbers for simplicity).

We define:
- Hidden state $x_t$ (starts at 0).  
- Input-dependent matrices $A(u_t), B(u_t), C(u_t)$.  
- A gating function $g(u_t)$.  

---

## 2. Equations
Mamba updates the hidden state as:  

$$x_{t+1} = g(u_t) \cdot (A(u_t) x_t + B(u_t) u_t) + (1 - g(u_t)) \cdot x_t$$  

$$y_t = C(u_t) x_t$$  

---

## 3. Step-by-Step Example

### Step 1: First token $u_1 = 2$
- Let $A(2) = 0.5, B(2) = 1.0, C(2) = 1.0, g(2) = 1$.  
- Update:  

$$x_1 = 1 \cdot (0.5 \cdot 0 + 1.0 \cdot 2) = 2$$  

- Output:  

$$y_1 = 1.0 \cdot 2 = 2$$  

---

### Step 2: Second token $u_2 = 5$
- Let $A(5) = 0.3, B(5) = 0.8, C(5) = 1.2, g(5) = 0.7$.  
- Update:  

$$x_2 = 0.7 \cdot (0.3 \cdot 2 + 0.8 \cdot 5) + 0.3 \cdot 2$$  

$$= 0.7 \cdot (0.6 + 4) + 0.6 = 0.7 \cdot 4.6 + 0.6 = 3.22 + 0.6 = 3.82$$  

- Output:  

$$y_2 = 1.2 \cdot 3.82 \approx 4.58$$  

---

### Step 3: Third token $u_3 = 3$
- Let $A(3) = 0.4, B(3) = 0.9, C(3) = 1.1, g(3) = 1$.  
- Update:  

$$x_3 = 1 \cdot (0.4 \cdot 3.82 + 0.9 \cdot 3) = 1.528 + 2.7 = 4.228$$  

- Output:  

$$y_3 = 1.1 \cdot 4.228 \approx 4.65$$  

---

## 4. Final Outputs
Sequence processed →  

$$y = [2, 4.58, 4.65]$$  


---

# 📌 Key Takeaway
- **Transformers:** Compare all tokens pairwise (quadratic cost).  
- **Mamba:** Sequentially updates a hidden state with **input-dependent parameters** and **gating**, yielding linear-time complexity.  
- This toy example shows how each token influences the evolving hidden state and output.

---

# 🐍 Mamba Toy Example Workflow

**Sequence Input:**  

$$u = [2, 5, 3]$$

---

## 🔄 Flow Diagram (Textual)

```
Input Tokens ---> [2] ---> [5] ---> [3]
                     |        |        |
                     v        v        v
                 State Update -----> Hidden State
                     |        |        |
                     v        v        v
                 Output y1   Output y2   Output y3
```

- Each token updates the **hidden state** using input-dependent matrices \(A(u_t), B(u_t), C(u_t)\).  
- The **gating function** decides how much of the state to update or keep.  
- Outputs are produced at each step:  
  - y1 = 2  
  - y2 ≈ 4.58  
  - y3 ≈ 4.65  

---

## 📊 Comparison with Transformer

**Transformer Workflow:**
```
Input Tokens ---> Attention Layer
   |   |   |
   v   v   v
Compare all tokens pairwise (O(n^2))
   |
   v
Output sequence embeddings
```

**Mamba Workflow:**
```
Input Tokens ---> Sequential State Updates (O(n))
   |
   v
Output sequence embeddings
```

---

## ✨ Key Difference
- **Transformer:** Every token talks to every other token → quadratic cost.  
- **Mamba:** Each token updates a running hidden state selectively → linear cost.  

---

👉 This diagram shows how Mamba processes inputs step by step, evolving a hidden state instead of doing pairwise comparisons like Transformers.  
