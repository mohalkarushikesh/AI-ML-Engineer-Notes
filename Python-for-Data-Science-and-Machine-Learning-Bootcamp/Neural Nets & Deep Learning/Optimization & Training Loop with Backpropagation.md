## 🔁 Optimization & Training Loop with Backpropagation

An optimization method such as **gradient descent** iteratively improves a neural network’s weights to minimize the loss.

---

### 🧠 Training Loop – 4 Core Steps

1. **Forward Pass**  
   Compute outputs from inputs **layer by layer**:
   $z = w \cdot x + b \quad \Rightarrow \quad a = \sigma(z)$

2. **Calculate Loss**  
   Compare predicted output  $\hat{y}$ with true output $y$ using a loss function $\mathcal{L}$.

3. **Backpropagation**  
   Starting from the output, **propagate the error backward** through the network using the **chain rule**.

4. **Gradient Descent Update**  
   Adjust each weight $w$ and bias $b$ to minimize the loss:
   $w := w - \eta \cdot \frac{\partial \mathcal{L}}{\partial w}$

---

## 🔄 Backpropagation – Deep Dive

Backpropagation answers:  
➡️ *How does the cost change with respect to each weight and bias in the network?*

Let the network have **\( L \)** layers.

---

### 🔹 Forward Step

Focusing on the last layers:

- **Weighted Sum**:
  $z^L = w^L a^{L-1} + b^L$
- **Activation**:
  $a^L = \sigma(z^L)$

---

### 🔹 Loss Function (e.g. MSE)

If cost function is:
$\mathcal{L} = \frac{1}{2} (a^L - y)^2$

Then we compute how sensitive this is to weight changes.

---

### 🔹 Error at Output Layer

The **error vector at layer L**:

$$
\delta^L = \nabla_a \mathcal{L} \odot \sigma'(z^L)
$$

If using MSE:
$\nabla_a \mathcal{L} = (a^L - y) \quad \Rightarrow \quad \delta^L = (a^L - y) \odot \sigma'(z^L)$

Here:
- $\odot$ = **Hadamard product** (element-wise multiplication)
- $\sigma'(z^L)$ = derivative of the activation function

---

### 🔹 Backpropagate the Error

For any hidden layer \( l \) from \( L-1 \) down to 1:

$\delta^l = \left( (w^{l+1})^T \delta^{l+1} \right) \odot \sigma'(z^l)$

- $(w^{l+1})^T$ : transpose of weights from next layer
- Multiplied with next layer's error
- Then element-wise multiplied with derivative of activation

---

### 🔹 Compute Gradients

Now compute gradients for updating:

- Weight gradient:
  $\frac{\partial \mathcal{L}}{\partial w^l} = \delta^l \cdot (a^{l-1})^T$

- Bias gradient:
  $\frac{\partial \mathcal{L}}{\partial b^l} = \delta^l$

---

### 🔧 Weight Update Rule

Using **gradient descent**, update weights and biases:

$$
w^l := w^l - \eta \cdot \frac{\partial \mathcal{L}}{\partial w^l}
$$

$$
b^l := b^l - \eta \cdot \frac{\partial \mathcal{L}}{\partial b^l}
$$

---

## ✨ Summary Flow

1. **Forward pass** through all layers
2. Compute **loss** using predicted and true values
3. Start at output and compute **$\delta^L$**
4. Use recurrence to find **$\delta^l$** for all layers
5. Compute gradients for all weights/biases
6. **Update parameters** using gradient descent

---
