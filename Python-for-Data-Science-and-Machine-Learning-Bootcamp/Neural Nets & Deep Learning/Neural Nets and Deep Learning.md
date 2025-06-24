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

- **Binary Cross-Entropy**:  
  $$\mathcal{L} = -[y \log(a) + (1 - y) \log(1 - a)]$$

- **Categorical Cross-Entropy**:  
  $$\mathcal{L} = -\sum_{i=1}^{C} y_i \log(p_i)$$

- **Mean Squared Error (MSE)**:  
  $$\mathcal{L} = \frac{1}{n} \sum_{i=1}^{n} (y_i - a_i)^2$$

---

#### 🔁 Backpropagation

Computes gradients using the **chain rule**:

$$
\frac{\partial \mathcal{L}}{\partial w}, \quad \frac{\partial \mathcal{L}}{\partial b}
$$

---

#### ⚙️ Gradient Descent

Weight update rule:

$$
w := w - \eta \cdot \frac{\partial \mathcal{L}}{\partial w}
$$

Where:
- $$\( \eta \)$$ = learning rate
- Step size determines convergence speed vs stability

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
