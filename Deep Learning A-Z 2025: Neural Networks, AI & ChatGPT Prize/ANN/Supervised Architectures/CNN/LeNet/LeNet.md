## 📘 LeNet Architecture (1998, Yann LeCun)

### 🔹 Key Features
- Designed for **handwritten digit recognition** (MNIST dataset).  
- One of the first successful CNNs, pioneering deep learning in computer vision.  
- Uses convolution + pooling layers followed by fully connected layers.  

---

### 🔹 Architecture

<img width="952" height="480" alt="lenet-min" src="https://github.com/user-attachments/assets/917a746d-ed2f-4a7c-be0d-af3d932e2804" />

***

## 1. Input Image

*   **Size:** `32 × 32 × 1`
    *   32 × 32 → height and width of the image
    *   1 → number of channels (grayscale image)

***

## 2. Convolutional Layer (C1)

The convolutional layer is defined by these hyperparameters:

| Parameter | Meaning              | Value |
| --------- | -------------------- | ----- |
| `n_c`     | Number of filters    | 6     |
| `F`       | Filter (kernel) size | 5 × 5 |
| `P`       | Padding              | 0     |
| `S`       | Stride               | 1     |

***

#### Filters (Kernels)

*   There are **6 filters**
*   Each filter has size:
        5 × 5 × 1
    *   5 × 5 → spatial size
    *   1 → depth must match input channels

Each filter slides across the image and produces **one feature map**.

***

#### Output Size Calculation

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

#### Trainable Parameters (Weights + Bias)

### Weights

*   Each filter has:
        5 × 5 × 1 = 25 weights
*   For 6 filters:
        25 × 6 = 150 weights

#### Bias

*   One bias per filter:
        6 biases

#### ✅ Total Trainable Parameters

    150 (weights) + 6 (biases) = 156

That’s why the diagram shows:

    5 × 5 × 1 × 6 + 6 = 156

***

#### Connections

![C1_Convolutional-Layer](https://github.com/user-attachments/assets/f6f4de71-703c-4257-b1c2-878ff94590ba)

## 3. **Pooling Layer (S2)**  
   - Average pooling (subsampling).  
   - Reduces spatial dimensions.  

![Pooling-Layer-](https://github.com/user-attachments/assets/053d1417-ff07-443b-a1ae-238329a0c117)


## 4. **Convolutional Layer (C3)**  
   - 16 filters of size 5×5.  
   - Produces 16 feature maps.  

![Convolutional-Layer-2](https://github.com/user-attachments/assets/70391eae-1a8a-4068-80ee-0ba03d1f277a)

## 5. **Pooling Layer (S4)**  
   - Average pooling again.  

![S4_Pooling-Laye](https://github.com/user-attachments/assets/1b4fda1c-eaa2-4066-9476-7207799b9207)

## 6. **Convolutional Layer (C5)**  
   
![C5_-Fully-Connected-laye](https://github.com/user-attachments/assets/8b216970-fa1b-4728-9eb0-8b6de5c73c88)

## 7. **Fully Connected Layers (C5)**  
   - Dense layers for classification.  

![f6_-Fully-Connected-Laye](https://github.com/user-attachments/assets/262bb9d8-2a54-44b9-b3de-b8e41bedb0fe)

## 8. **Output Layer**  
   - Softmax classifier (e.g., 10 classes for digits 0–9).  

![file](https://github.com/user-attachments/assets/f39668e5-fc87-42da-990e-ef17de77dca8)

---

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

