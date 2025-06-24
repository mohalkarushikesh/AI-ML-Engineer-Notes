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
  $$y = x_1w_1 + x_2w_2$$

- Add bias term \( b \):  
  $$y = x_1w_1 + x_2w_2 + b$$  
  This allows shifting the decision boundary away from the origin.

- If:  
  $$b = -10,\ z = xw + b$$  
  Then output won’t activate unless \( xw > 10 \), hence the term **bias**.

<img src="![image](https://github.com/user-attachments/assets/3f0d1aac-71c3-4cc4-bc24-0c1990bd4ba4)" alt="alt text" style="width:50%; height:250;">

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

Used to introduce non-linearity and control outputs:

| Function    | Formula / Logic                          | Output Range | Notes                                 |
|-------------|-------------------------------------------|---------------|----------------------------------------|
| **Step**    | Outputs 0 or 1                            | 0 or 1        | Not commonly used in modern networks   |
| **Sigmoid** | \( \frac{1}{1 + e^{-z}} \)                | (0, 1)        | Good for binary classification         |

<img src="image-2.png" alt="alt text" style="width:50%; height:250;">

| **Tanh**    | \( \tanh(z) \)                            | (-1, 1)       | Zero-centered alternative to sigmoid   |
<img src="imag-3.png" alt="alt text" style="width:50%; height:250;">
| **ReLU**    | \( \max(0, z) \)                          | [0, ∞)        | Prevents vanishing gradient, fast      |
| **Softmax** | \( \frac{e^{z_i}}{\sum_k e^{z_k}} \)      | (0, 1), sum=1 | Used for multi-class classification    |

---

### **5. Multi-Class Classification**

#### **Non-Exclusive Classes** (multi-label)

- Each data point can belong to multiple categories (e.g. “beach”, “family”, “vacation”):
  

<img src="image-5.png" alt="alt text" style="width:50%; height:250;">
<img src="image-6.png" alt="alt text" style="width:50%; height:250;">
<img src="image-7.png" alt="alt text" style="width:50%; height:250;">

#### **Mutually Exclusive Classes** (single-label)

- Each input is assigned only one class label (e.g. “dog” **or** “cat”, not both)
- **Softmax Activation** ensures only one class is assigned with highest probability  
  Example output: `[Red: 0.1, Green: 0.6, Blue: 0.3]`

<img src="image-4.png" alt="alt text" style="width:50%; height:250;">

#### **One-Hot Encoding**:
- Converts labels into vectors  
  Green → `[0, 1, 0]`

---

### **6. Training the Network**

- **Forward Propagation**: Input flows through each layer
- **Cost/Loss Functions**:
  - Binary Cross-Entropy
  - Categorical Cross-Entropy
  - Mean Squared Error (MSE), etc.
- **Backpropagation**: Adjusts weights based on loss gradients
- **Gradient Descent**: Optimization to minimize error

---

### **7. TensorFlow & Keras Implementation**

Use **TensorFlow 2.x** and **Keras** for creating and training neural networks.

#### Key Components:
- **Layers**: `Dense`, `Dropout`, `Activation`
- **Optimizers**: `adam`, `sgd`, `rmsprop`
- **Metrics**: `accuracy`, `loss`
- **Callbacks**: `EarlyStopping`, `ModelCheckpoint`

#### Visualization:
- **TensorBoard** – for tracking metrics, graphs, and more.

---

### ⚙️ Example (Keras Classification)

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
