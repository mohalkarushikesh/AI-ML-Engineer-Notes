# ⚡ Optimizers in Machine Learning — All in One Notes

## 🔹 What is an Optimizer?
- An **optimizer** is an algorithm that updates model parameters (weights) to minimize the loss function.  
- Works with **gradient descent**: adjusts weights in the direction that reduces error.  
- Goal: faster convergence, better generalization, and stability.

---

## 📈 Common Optimizers

| Optimizer | Update Rule | Pros | Cons | Best Use |
|-----------|-------------|------|------|----------|
| **Gradient Descent (GD)** | $w = w - \eta \cdot \nabla L(w)$ | Simple, foundational | Slow, stuck in local minima | Small datasets |
| **Stochastic Gradient Descent (SGD)** | Updates per sample | Faster, less memory | Noisy updates | Large datasets |
| **Mini-Batch SGD** | Updates per batch | Balance speed & stability | Needs tuning batch size | Deep learning |
| **Momentum** | $v_t = \beta v_{t-1} + \eta \nabla L(w)$ | Accelerates, reduces oscillation | Extra hyperparameter $\beta$ | ConvNets |
| **Nesterov Accelerated Gradient (NAG)** | Lookahead momentum | Faster convergence | More complex | Deep nets |
| **Adagrad** | $w = w - \frac{\eta}{\sqrt{G_t}} \nabla L(w)$ | Adapts learning rate per parameter | Learning rate shrinks too much | Sparse data |
| **RMSProp** | $w = w - \frac{\eta}{\sqrt{E[\nabla^2]}} \nabla L(w)$ | Fixes Adagrad decay | Needs tuning | RNNs |
| **Adam (Adaptive Moment Estimation)** | Combines Momentum + RMSProp | Fast, widely used | Can overfit, needs tuning | Most DL tasks |
| **Nadam** | Adam + Nesterov | Better convergence | More complex | NLP, seq models |
| **AdaMax** | Variant of Adam using infinity norm | Stable | Less common | Large embeddings |
| **AMSGrad** | Adam with non-decreasing step sizes | Avoids convergence issues | Slower | Theoretical guarantees |

---

## 📐 Key Formulas

- **Gradient Descent**:  
  $$w_{t+1} = w_t - \eta \cdot \nabla L(w_t)$$  

- **Momentum**:  
  $$v_t = \beta v_{t-1} + \eta \cdot \nabla L(w_t)$$  
  $$w_{t+1} = w_t - v_t$$  

- **Adagrad**:  
  $$w_{t+1} = w_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \cdot \nabla L(w_t)$$  

- **RMSProp**:  

$$
E[g^2]_t = \gamma E[g^2]_{t-1} + (1-\gamma) g_t^2
$$

$$
w_{t+1} = w_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} g_t
$$  

- **Adam**:  
  $$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$  
  $$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$  

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$
  
$$
w_{t+1} = w_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
$$  

---

## 🧪 Choosing the Right Optimizer

| Scenario | Recommended Optimizer |
|----------|------------------------|
| Small dataset, simple model | Gradient Descent / SGD |
| Large dataset | Mini-Batch SGD |
| Deep CNNs | Momentum / Adam |
| RNNs / Sequence models | RMSProp / Adam / Nadam |
| Sparse features | Adagrad |
| Imbalanced / unstable training | AMSGrad |

---

## ✅ Summary
- **SGD** is the baseline.  
- **Momentum/NAG** accelerate convergence.  
- **Adagrad/RMSProp** adapt learning rates.  
- **Adam/Nadam** combine best of both worlds → most popular.  
- Specialized variants (AdaMax, AMSGrad) fix convergence or stability issues.  

---
