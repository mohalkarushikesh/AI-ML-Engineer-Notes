## 🧠 Neural Networks & Deep Learning – Full Notes

### **1. From Biology to Perceptron**

- Inspired by **biological neurons**:
  - **Dendrites** = input signals  
  - **Nucleus** = processing  
  - **Axon** = output signal

- **Perceptron Model** (1958, *Frank Rosenblatt*):
  - A simplified model of a neuron: receives weighted inputs, sums them, applies an activation function, and produces an output.
  - Initially hoped to learn, make decisions, and even translate languages.

- In **1969**, *Marvin Minsky* and *Seymour Papert* published *"Perceptrons"*, highlighting its limitations (especially with non-linear problems). This contributed to the **first AI Winter**.

---

### **2. Perceptron Math Refresher**

- Without weights:  
  $$y = x_1 + x_2$$

- Add weights:  
  $$y = x_1 w_1 + x_2 w_2$$

- Add bias term \( b \):  
  $$y = x_1 w_1 + x_2 w_2 + b$$  
  This allows shifting the decision boundary away from the origin.

- If:  
  $$b = -10,\quad z = xw + b$$  
  Then output won’t activate unless \( xw > 10 \), hence the term **bias**.

<img src="https://github.com/user-attachments/assets/1b874724-b211-4054-8e58-67cc60e56081" alt="alt text" style="width:50%; height:250;">

---

### **3. Neural Network Structure**

- **Multilayer Perceptron (MLP)**:
  - **Input Layer** – receives raw data
  - **Hidden Layers** – process patterns/features
  - **Output Layer** – returns final prediction/output

- A network becomes **deep** when it has **2 or more hidden layers**.

- **Universal Approximation Theorem**:  
  Neural networks can approximate **any continuous convex function**, given sufficient neurons and layers.

---

### **4. Activation Functions**

Used to introduce non-linearity and control outputs

| Function    | Formula                          | Output Range | Notes                                 |
|-------------|-----------------------------------|---------------|----------------------------------------|
| **Step**    | Outputs 0 or 1                    | 0 or 1        | Not commonly used in modern networks   |
| **Sigmoid** | $\frac{1}{1 + e^{-z}}$            | (0, 1)        | Good for binary classification         |

<img src="https://github.com/user-attachments/assets/4b06ff6b-08b0-41df-aa48-d7bbe21f3db7" alt="alt text" style="width:50%; height:250;">

| **Tanh**    | $\tanh(z)$                        | (-1, 1)       | Zero-centered alternative to sigmoid   |

<img src="https://github.com/user-attachments/assets/b156b57a-6813-4a05-8b10-26db2f12c49f" alt="alt text" style="width:50%; height:250;">

| **ReLU**    | $\max(0, z)$                      | [0, ∞)        | Prevents vanishing gradient, fast      |
| **Softmax** | $\frac{e^{z_i}}{\sum_k e^{z_k}}$  | (0, 1), sum=1 | Used for multi-class classification    |


---

### **5. Multi-Class Classification**

#### **Non-Exclusive Classes** (multi-label)

- Each data point can belong to multiple categories (e.g. “beach”, “family”, “vacation”):

<img src="https://github.com/user-attachments/assets/10e18dde-8572-4c54-b143-bdaa9124aaac" alt="alt text" style="width:50%; height:250;">
<img src="https://github.com/user-attachments/assets/b80c6054-a8fc-434c-8603-e3a7ba3b8a45" alt="alt text" style="width:50%; height:250;">
<img src="https://github.com/user-attachments/assets/01fe0a25-2370-4421-8a21-a6dd00e6163b" alt="alt text" style="width:50%; height:250;">

#### **Mutually Exclusive Classes** (single-label)

- Each input is assigned only one class label
- **Softmax Activation** ensures only one class is assigned the highest probability  
  Example: `[Red: 0.1, Green: 0.6, Blue: 0.3]`

<img src="https://github.com/user-attachments/assets/24d9c320-1b0d-4aa4-b1a6-32a24b490ef8" alt="alt text" style="width:50%; height:250;">

#### **One-Hot Encoding**:
- Converts labels into vectors  
  Green → `[0, 1, 0]`

---

### **6. Training the Network**

#### 🧠 Forward Propagation

- Each layer computes:

  $$z = w \cdot x + b$$
  $$a = \sigma(z)$$

- Final prediction:

  $$\hat{y} = a^{(L)}$$

---

#### 📉 Loss / Cost Functions

### 🔹 **1. Binary Cross-Entropy**

$$\mathcal{L} = -[y \log(a) + (1 - y) \log(1 - a)]$$

#### ✅ When to use:
- **Binary classification tasks**, where the output is 0 or 1 (e.g., spam or not spam).
- The model outputs a **probability** \( a \in (0, 1) \) using a **sigmoid** activation.

#### 🧠 Intuition:
- If the true label \( y = 1 \), the loss becomes \( -\log(a) \)
- If \( y = 0 \), the loss becomes \( -\log(1 - a) \)
- The function **punishes incorrect confident predictions** (e.g., predicting 0.01 when the truth is 1)

#### 📌 Characteristics:
- Output is always non-negative
- Highly sensitive to prediction confidence
- As prediction approaches true value, loss tends toward 0

---

### 🔹 **2. Categorical Cross-Entropy**

$$\mathcal{L} = -\sum_{i=1}^{C} y_i \log(p_i)$$

#### ✅ When to use:
- **Multi-class classification** where only **one class is correct**
- Model outputs a **vector of probabilities** across \( C \) classes using **Softmax** activation

#### 🧠 Intuition:
- Only one \( y_i \) is 1 (true class), rest are 0
- So the sum reduces to: \( -\log(\text{probability of the correct class}) \)
- Encourages the model to assign high probability to the correct class

#### 📌 Characteristics:
- Assumes **one-hot encoding** of labels
- Loss decreases as predicted probability of true class increases
- Generalization of binary cross-entropy for >2 classes

---

### 🔹 **3. Mean Squared Error (MSE)**

$$\mathcal{L} = \frac{1}{n} \sum_{i=1}^{n} (y_i - a_i)^2$$

#### ✅ When to use:
- **Regression problems**, where the output is a continuous value
- You want to minimize the **average squared difference** between prediction and ground truth

#### 🧠 Intuition:
- Penalizes large errors more heavily than small ones
- Squaring the error keeps all values positive and amplifies larger deviations

#### 📌 Characteristics:
- Smooth and easy to compute
- Sensitive to outliers (because of the squared term)
- Not ideal for classification (cross-entropy performs better there)

---

Each of these loss functions plays a key role in shaping how a neural network learns. Choosing the **right one for your problem type** is crucial for successful model training.

---

#### 🔁 Backpropagation

## 🔁 What Is Backpropagation?

Backpropagation is a **recursive algorithm** used during training to compute the **gradient of the loss function** with respect to all the weights and biases in the network.

Think of it as **error attribution**: it tells each neuron how much it contributed to the final prediction error, so we can update the weights accordingly.

---

## 🧠 Intuition Before Math

1. **Forward Pass**: Compute outputs from inputs layer by layer.
2. **Compute Loss**: Compare prediction $\hat{y}$ with true output $y$ using a loss function $\mathcal{L}$.
3. **Backward Pass**: Starting from the output, propagate the error *back* through the network using the chain rule.
4. **Update Weights**: Adjust each weight using **gradient descent**.

---

## 📐 Notation Setup

Let’s define a few basics:

- **Activations in layer \( l \):**  
  $$
  a^l
  $$

- **Weights connecting layer \( l-1 \) to \( l \):**  
  $$
  w^l
  $$

- **Bias vector for layer \( l \):**  
  $$
  b^l
  $$

- **Weighted input to layer \( l \):**  
  $$
  z^l = w^l a^{l-1} + b^l
  $$

- **Activation function (e.g., ReLU, sigmoid):**  
  $$
  \sigma(z)
  $$

- **Loss function:**  
  $$
  \mathcal{L}(a^L, y)
  $$

---

## 🔬 Step-by-Step Math

### 1. **Forward Propagation**

For each layer \( l \):
- Weighted sum:  
  $$
  z^l = w^l a^{l-1} + b^l
  $$
- Activation:  
  $$
  a^l = \sigma(z^l)
  $$

---

### 2. **Compute Output Layer Error**

At the output layer \( L \), compute:
- Error:  
  $$
  \delta^L = \nabla_a \mathcal{L}(a^L, y) \odot \sigma'(z^L)
  $$
Where:

- Derivative of loss wrt output activations
$$
a_a \mathcal{L} 
$$

- Element-wise multiplication
$$ 
\odot 
$$ 
- Derivative of the activation function at layer \( L \)
$$
\sigma'(z^L) 
$$

For **mean squared error**:
$$
\delta^L = (a^L - y) \odot \sigma'(z^L)
$$

---

### 3. **Backpropagate the Error**

For any hidden layer $l$ (from $L-1$ to 1):
$$
\delta^l = ((w^{l+1})^T \delta^{l+1}) \odot \sigma'(z^l)
$$

- Multiply the upstream error by the transpose of the weights,
- Then element-wise multiply with the derivative of the activation at that layer.

---

### 4. **Compute Gradients**

Now that we have $\delta^l$, compute:

- Gradient w.r.t. weights:
  $$
  \frac{\partial \mathcal{L}}{\partial w^l} = \delta^l (a^{l-1})^T
  $$

- Gradient w.r.t. bias:
  $$
  \frac{\partial \mathcal{L}}{\partial b^l} = \delta^l
  $$

---

### 5. **Update Parameters**

Using gradient descent:

$$
w^l := w^l - \eta \cdot \frac{\partial \mathcal{L}}{\partial w^l}
$$

$$
b^l := b^l - \eta \cdot \frac{\partial \mathcal{L}}{\partial b^l}
$$

Where $\eta$ is the learning rate.

---

## 🔁 Loop During Training

For each batch of data:
- Perform **forward pass**
- Calculate **loss**
- Run **backpropagation**
- Apply **gradient descent updates**

Repeat over many **epochs**.

---

## 🔎 Final Notes

- Backpropagation uses **chain rule** to efficiently compute gradients
- It works with any differentiable activation/loss function
- With large networks, frameworks like TensorFlow/PyTorch handle this **automatically** via autodiff

---

### ⚙️ Gradient Descent – In Depth

Gradient Descent is an **optimization algorithm** used to minimize a function—most often the **loss function** in machine learning and deep learning.

In our context, the function we want to minimize is the **loss** $\mathcal{L}$, which measures how far off the model’s predictions are from the true values.

---

### 🔁 The Update Rule

$$w := w - \eta \cdot \frac{\partial \mathcal{L}}{\partial w}$$

Let’s break this down:

| Term | Meaning |
|------|---------|
| **$w$** | The current weight (parameter) of the model |
| **$\frac{\partial \mathcal{L}}{\partial w}$** | The **gradient**: how much the loss changes with respect to the weight |
| **$\eta$** | The **learning rate**: a small positive scalar that controls how big a step we take |
| **$w :=$** | Notation for updating the weight (i.e., “replace $w$ with this new value”) |

---

### 📉 Why Subtract the Gradient?

Imagine you're standing on a mountain in the fog, trying to reach the bottom. You can’t see it—but you can feel the slope beneath your feet. You take a small step in the direction that goes *downhill most steeply*. This is exactly what subtracting the gradient does:  
- The gradient points in the **direction of steepest ascent**,  
- So we **subtract** it to move *downhill* toward the minimum of the loss.

---

### 🧭 Role of Learning Rate ($\eta$)

- If $\eta$ is **too small**, learning is **very slow**—like taking baby steps.
- If $\eta$ is **too large**, you might **overshoot** the minimum—or even diverge and make things worse.
- The ideal $\eta$ gives a smooth, steady descent to the lowest point.

---

### 📈 Intuition With a Plot

If you plotted your loss $\mathcal{L}(w)$ against the weight $w$:

- You’d see a curve—maybe a bowl shape.
- The goal is to reach the **bottom** of the bowl where loss is minimized.
- Each update nudges your position slightly closer to the bottom.

---

### 💡 Final Thoughts

Gradient descent is powerful but not magic—it relies on careful tuning of the learning rate and may get stuck in local minima in complex models. That's why enhancements like **momentum**, **RMSProp**, and **Adam** were developed.

---

#### 🤖 Adam Optimizer

Adaptive technique combining:
- Momentum
- Per-parameter learning rates

**Reference**: *Kingma & Ba, 2015*

---

### ✅ Summary Table

| Step | Description |
|------|-------------|
| Forward Propagation | Compute outputs layer by layer |
| Loss Function | Quantify prediction error |
| Backpropagation | Compute gradients |
| Gradient Descent | Update weights to reduce loss |
| Adam Optimizer | Efficient, adaptive optimization |

---

### **7. TensorFlow & Keras Example**

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(input_dim,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```
