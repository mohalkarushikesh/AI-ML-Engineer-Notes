# 📘 LeNet-5 Architecture 

<img width="952" height="480" alt="lenet-min" src="https://github.com/user-attachments/assets/1185edb2-379d-4f06-b1b5-ebc657e4a3ba" />

---

## 1️⃣ Input Layer

**Input Image Size:**

```
32 × 32 × 1
```

### Why 32×32 and not 28×28?

* MNIST digits are **28×28**
* LeNet pads them to **32×32**
* Reason:

  * Allows **more spatial room** for convolutions
  * Avoids losing border information early

---

## 2️⃣ Convolutional Layer — **C1** ✅ 

![C1_Convolutional-Layer](https://github.com/user-attachments/assets/204db142-de6c-48d4-aa99-f0026ee0edfa)


##### Convolutional Layer Settings

The convolutional layer is defined by these hyperparameters:

| Parameter | Meaning              | Value |
| --------- | -------------------- | ----- |
| `n_c`     | Number of filters    | 6     |
| `F`       | Filter (kernel) size | 5 × 5 |
| `P`       | Padding              | 0     |
| `S`       | Stride               | 1     |

***

##### Filters (Kernels)

*   There are **6 filters**
*   Each filter has size:
        5 × 5 × 1
    *   5 × 5 → spatial size
    *   1 → depth must match input channels

Each filter slides across the image and produces **one feature map**.

***

##### Output Size Calculation

The output spatial size is computed using:

$$
\text{Output size} = \frac{N - F + 2P}{S} + 1
$$

Where:

*   $$N = 32$$
*   $$F = 5$$
*   $$P = 0$$
*   $$S = 1$$

$$
\frac{32 - 5 + 0}{1} + 1 = 28
$$

✅ **Output size:** `28 × 28 × 6`

*   28 × 28 → height and width
*   6 → one feature map per filter

***

#####  Trainable Parameters (Weights + Bias)

##### Weights

*   Each filter has:
        5 × 5 × 1 = 25 weights
*   For 6 filters:
        25 × 6 = 150 weights

##### Bias

*   One bias per filter:
        6 biases

##### ✅ Total Trainable Parameters

    150 (weights) + 6 (biases) = 156

That’s why the diagram shows:

    5 × 5 × 1 × 6 + 6 = 156

***

## 3️⃣ Pooling Layer — **S2 (Subsampling Layer)**

![Pooling-Layer-](https://github.com/user-attachments/assets/053d1417-ff07-443b-a1ae-238329a0c117)

### Input to S2

```
28 × 28 × 6
```

### Pooling Hyperparameters

| Parameter    | Value               |
| ------------ | ------------------- |
| Pool size    | 2 × 2               |
| Stride       | 2                   |
| Pooling type | **Average pooling** |

---

### Output Size Calculation

[
\frac{28}{2} = 14
]

```
14 × 14 × 6
```

---

### What Actually Happens?

For **each 2×2 block**:

* Take the **average**
* Multiply by a **trainable weight**
* Add a **trainable bias**
* Apply activation (tanh)

⚠️ Important:
Unlike modern pooling, **LeNet pooling has parameters**.

---

### Trainable Parameters in S2

* One weight + one bias **per feature map**
* Total feature maps = 6

```
6 weights + 6 biases = 12 parameters
```

📌 Purpose:

* Reduce spatial size
* Reduce noise
* Make features more robust to small shifts

---

## 4️⃣ Convolutional Layer — **C3**

![Convolutional-Layer-2](https://github.com/user-attachments/assets/70391eae-1a8a-4068-80ee-0ba03d1f277a)

### Input to C3

```
14 × 14 × 6
```

---

### Hyperparameters

| Parameter   | Value |
| ----------- | ----- |
| Filters     | 16    |
| Kernel size | 5 × 5 |
| Stride      | 1     |
| Padding     | 0     |

---

### Output Size

[
\frac{14 - 5}{1} + 1 = 10
]

```
10 × 10 × 16
```

---

### ⚠️ UNIQUE LeNet FEATURE: **Partial Connectivity**

Unlike modern CNNs:

* **Not every filter connects to all 6 input maps**
* Some filters see:

  * 3 maps
  * Some see 4
  * Some see all 6

📌 Why?

* Reduce parameters
* Force diversity of learned features
* Hardware limitations (1998!)

---

### Parameter Calculation (Conceptual)

Each filter:

```
5 × 5 × (connected input maps)
```

Total parameters ≈ **1,516**

📌 What C3 learns:

> **Combinations of edges → shapes → digit parts**

---

## 5️⃣ Pooling Layer — **S4**

![S4\_Pooling-Laye](https://github.com/user-attachments/assets/1b4fda1c-eaa2-4066-9476-7207799b9207)

### Input

```
10 × 10 × 16
```

### Pooling Setup

* Pool size: 2×2
* Stride: 2
* Average pooling + trainable scale & bias

---

### Output

```
5 × 5 × 16
```

---

### Trainable Parameters

* One weight + bias per feature map

```
16 × 2 = 32 parameters
```

📌 Role:

* Further spatial compression
* Keep **semantic meaning**
* Increase translation invariance

---

## 6️⃣ Convolutional Layer — **C5** (Acts like Fully Connected)

![C5\_-Fully-Connected-laye](https://github.com/user-attachments/assets/8b216970-fa1b-4728-9eb0-8b6de5c73c88)

### Input

```
5 × 5 × 16
```

### Filter Size

```
5 × 5 × 16
```

⚠️ IMPORTANT:

* Kernel size equals input size
* Output spatial dimension becomes **1×1**

---

### Number of Filters

```
120
```

### Output

```
1 × 1 × 120
```

(Usually written as just **120 neurons**)

---

### Trainable Parameters

Each filter:

```
5 × 5 × 16 = 400 weights
```

Total:

```
400 × 120 + 120 biases = 48,120
```

📌 What C5 learns:

> **High-level digit concepts** (loops, strokes, shapes)

---

## 7️⃣ Fully Connected Layer — **F6**

![f6\_-Fully-Connected-Laye](https://github.com/user-attachments/assets/262bb9d8-2a54-44b9-b3de-b8e41bedb0fe)

### Input

```
120 neurons
```

### Output

```
84 neurons
```

---

### Parameters

```
120 × 84 + 84 = 10,164
```

📌 Purpose:

* Combine all learned features
* Prepare for classification

Activation: **tanh**

---

## 8️⃣ Output Layer — **Softmax**

![file](https://github.com/user-attachments/assets/f39668e5-fc87-42da-990e-ef17de77dca8)

### Input

```
84
```

### Output

```
10 neurons (digits 0–9)
```

### Parameters

```
84 × 10 + 10 = 850
```

---

### Softmax Formula

```math
P(y=i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}
```

📌 Produces:

* Class probabilities
* Final digit prediction

---

## 🔢 Total Parameters (Approx)

| Layer     | Parameters |
| --------- | ---------- |
| C1        | 156        |
| S2        | 12         |
| C3        | 1,516      |
| S4        | 32         |
| C5        | 48,120     |
| F6        | 10,164     |
| Output    | 850        |
| **Total** | **~60K**   |

⚠️ Compare:

* LeNet: ~60K params
* Modern CNNs: **millions to billions**

---

## 🧠 Why LeNet Was Revolutionary

* Introduced **end-to-end learning**
* Convolution + pooling concept
* Weight sharing
* Translation invariance
* Inspired **AlexNet → VGG → ResNet**

---

## 🔑 One-Line Mental Model

> **Edges → Shapes → Parts → Digits**


### 🔹 Characteristics
- **Activation Function:** Sigmoid or tanh (ReLU wasn’t popular yet).  
- **Pooling:** Average pooling (modern CNNs use max pooling).  
- **Parameters:** Much fewer compared to modern CNNs.  
- **Strength:** Showed CNNs could outperform traditional methods in vision tasks.  

---

### 🔹 Applications
- Handwritten digit recognition (MNIST).  
- Early document recognition systems.  
- Foundation for modern CNNs like AlexNet, VGG, ResNet.  

---

✅ **In short:** LeNet is the pioneering CNN that introduced convolution + pooling + fully connected layers for image recognition, laying the groundwork for modern deep learning in computer vision.

---

✅ **This flow shows:**  
- Input image → convolution → pooling → deeper convolution → pooling → fully connected layers → output.  
- It’s the classic CNN pipeline that inspired modern architectures like AlexNet, VGG, and ResNet.  

