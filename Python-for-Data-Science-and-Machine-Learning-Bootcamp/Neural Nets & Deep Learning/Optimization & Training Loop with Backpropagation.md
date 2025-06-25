## 🔄 Gradient Descent & Backpropagation – Simplified Explanation

An **optimization method** like gradient descent helps us **improve a neural network** by adjusting its weights so it makes better predictions.

---

### 🔁 Loop During Training

Each time the model trains on data, it repeats these 4 steps:

1. **Forward Pass**  
   Pass the input through the network one layer at a time and calculate the output.

2. **Calculate Loss**  
   Compare the model’s prediction ($\hat{y}$) with the actual label ($y$) using a loss function like Mean Squared Error.

3. **Backpropagation**  
   Figure out how much each weight contributed to the error and how to fix it, starting from the output and moving backward.

4. **Gradient Descent Update**  
   Adjust the weights using those gradients to make the predictions better next time.

---

## 🔁 Backpropagation – Step by Step

Let’s say your network has **L layers**.

### 🧱 Notation Setup

- $z = w \cdot x + b$: weighted sum  
- $a = \sigma(z)$: apply activation function  
- $\sigma$: any activation function (like ReLU or sigmoid)

So for the last layer:

- $z^L = w^L \cdot a^{L-1} + b^L$  
- $a^L = \sigma(z^L)$  
- Cost: $\mathcal{L} = (a^L - y)^2$

We want to know:  
👉 *How much does the loss change when we change the weights?*

That’s where gradients come in.

---

### 🧠 Step 1: Forward Pass (again)

For each layer, do:
- $z = w \cdot x + b$
- $a = \sigma(z)$  
This flows from the input all the way to the output.

---

### 📉 Step 2: Compute the Error at Output

We define the **error at the final layer** as:

- $\delta^L = (a^L - y) \odot \sigma'(z^L)$  
   (Here ⊙ means element-wise multiplication)

This tells us:  
👉 *How much did the final layer’s output mess up the prediction?*

---

### 🔁 Step 3: Backpropagate the Error

Now we move backward through the network to figure out each layer’s error.

For every earlier layer:
- $\delta^l = (w^{l+1})^T \cdot \delta^{l+1} \odot \sigma'(z^l)$  
  - $(w^{l+1})^T$ is the **transpose** of weights from the next layer
  - $\sigma'(z^l)$ is the derivative of the activation function

This helps us calculate how much *each neuron in this layer* contributed to the final error.

---

### 🧮 Step 4: Compute Gradients

Now that we have errors for each layer, we can calculate:

- $\frac{\partial \mathcal{L}}{\partial w^l} = \delta^l \cdot (a^{l-1})^T$
- $\frac{\partial \mathcal{L}}{\partial b^l} = \delta^l$

These tell us exactly how to adjust the weights and biases.

---

### 🔧 Step 5: Update Parameters

Now apply the updates using **gradient descent**:

- $w^l := w^l - \eta \cdot \frac{\partial \mathcal{L}}{\partial w^l}$
- $b^l := b^l - \eta \cdot \frac{\partial \mathcal{L}}{\partial b^l}$

Where:
- $\eta$ is the **learning rate**, which controls how big of a step we take

---
