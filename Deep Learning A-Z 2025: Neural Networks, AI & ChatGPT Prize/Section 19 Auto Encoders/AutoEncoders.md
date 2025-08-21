## 🔍 Autoencoders: Overview

- **Autoencoders** are neural networks designed to learn efficient representations of data by compressing and reconstructing it.
- They consist of:
  - **Encoder**: Maps input data to a lower-dimensional latent space.
  - **Decoder**: Reconstructs the original data from the latent representation.
- The goal is to minimize **reconstruction error** using loss functions like:
  - **Mean Squared Error (MSE)**
  - **Binary Cross-Entropy (BCE)**

### ✨ Applications:
- Noise reduction
- Feature extraction
- Anomaly detection
- Dimensionality reduction

<img width="828" height="546" alt="image" src="https://github.com/user-attachments/assets/33773610-2a83-403c-afb2-238bf0347036" />

🔗 Additional Reading:  
[Neural Networks Are Impressively Good at Compression – Malte Skarupke (2016)](https://probablydance.com/2016/04/30/neural-networks-are-impressively-good-at-compression/)

---

## ⚙️ Biases in Autoencoders

- **Bias terms** allow neurons to shift activation independently of input.
- They help the network better fit data by adjusting outputs flexibly.
- Present in both encoder and decoder layers.

---

## 🏋️ Training Autoencoders

- Trained using **backpropagation** and optimizers like **Adam** or **SGD**.
- Loss is computed between input and reconstructed output.
- Regularization techniques (e.g., dropout, weight decay) help prevent overfitting.

<img width="1216" height="653" alt="image" src="https://github.com/user-attachments/assets/4c9a9b63-f4e8-406e-9658-737d207964ea" />

🔗 Additional Reading:  
[Building Autoencoders in Keras – François Chollet (2016)](https://blog.keras.io/building-autoencoders-in-keras.html)

---

## 🔻 Undercomplete Autoencoders

- **Undercomplete layers** have fewer neurons than the input layer.
- They force the model to learn compressed, abstract representations.
- Prevents the network from simply copying the input.

### ✅ Benefits:
- Promotes generalization
- Ideal for dimensionality reduction
- Helps uncover latent structure in data

---

## 🔺 Overcomplete Autoencoders

### 🧠 What Are They?

- Hidden layers with **more neurons than the input layer**.
- Contrary to undercomplete models, they expand the input space.

### 🧱 Structure:
- Input: e.g., 784 pixels (MNIST)
- Hidden: e.g., 1024 or 2048 neurons
- Output: Same size as input

---

### 🎯 Why Use Overcomplete Layers?

- Learn **richer, more complex representations**
- Enhance **denoising** capabilities
- Improve **feature extraction** in unsupervised tasks
- Increase **robustness** to missing/corrupted data

---

### ⚠️ Risks and Challenges

- **Overfitting** due to excess capacity
- **Redundant features** may be learned
- **Training complexity** increases

---

### 🛡️ Regularization Techniques

| Technique         | Purpose                                        |
|------------------|------------------------------------------------|
| **Sparsity**      | Forces most hidden units to remain inactive   |
| **Denoising**     | Trains on noisy inputs to improve robustness  |
| **Contractive**   | Penalizes sensitivity to input changes        |
| **Dropout**       | Randomly disables neurons during training     |

---

### 🧪 PyTorch Example: Overcomplete Autoencoder

```python
class OvercompleteAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
```

---

### 🧭 When to Use Overcomplete Autoencoders

- **Image denoising**
- **Anomaly detection**
- **Pretraining for deep networks**
- **Learning robust features from unlabeled data**

---

## 🧬 Variants of Autoencoders

| Type                     | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| **Sparse Autoencoders**   | Enforce sparsity in hidden units to learn distinct features                |
| **Denoising Autoencoders**| Train on corrupted inputs to reconstruct clean outputs                     |
| **Contractive Autoencoders**| Penalize sensitivity to input changes to improve robustness             |
| **Stacked Autoencoders**  | Multiple layers of encoders/decoders for hierarchical feature learning     |
| **Deep Autoencoders**     | Deep architectures for capturing complex data patterns                     |

---

## 🔍 What Is a Sparse Autoencoder?

A **Sparse Autoencoder (SAE)** is a type of autoencoder that introduces a **sparsity constraint** on the hidden units. Unlike standard autoencoders that may activate all hidden neurons, SAEs encourage only a **small subset** of neurons to be active at any time.

This leads to:
- **Compact representations**
- **Better generalization**
- **Feature disentanglement**

---

## 🧱 Architecture Overview

- **Input Layer**: Raw data (e.g., image pixels, tabular features)
- **Hidden Layer**: More neurons than input (often overcomplete)
- **Sparsity Constraint**: Applied to hidden activations
- **Output Layer**: Reconstructed input

---

## 🧮 Objective Function

The loss function combines:
1. **Reconstruction Loss**: Measures how well the output matches the input  
   $$L_{\text{recon}} = \|X - \hat{X}\|^2$$

2. **Sparsity Penalty**: Encourages hidden activations to be close to a target sparsity level  
   $$L_{\text{sparse}} = \lambda \cdot \sum_{j=1}^{n} \text{KL}(\rho \| \hat{\rho}_j)$$  
   - ρ: Desired average activation (e.g., 0.05)
   - \( \hat{\rho}_j \): Actual average activation of neuron j
   - KL: Kullback-Leibler divergence

3. **Total Loss**:  
   $$L = L_{\text{recon}} + L_{\text{sparse}}$$

---

## 🛠️ Techniques to Enforce Sparsity

| Method              | Description                                                                 |
|---------------------|------------------------------------------------------------------------------|
| **KL Divergence**    | Penalizes deviation from target activation level                            |
| **L1 Regularization**| Encourages weights to be small, leading to sparse activations               |
| **Dropout**          | Randomly disables neurons during training to reduce reliance on all units   |

---

## 🧪 Training Process

1. **Initialization**: Random or pre-trained weights
2. **Forward Pass**: Input → Encoder → Decoder → Output
3. **Loss Calculation**: Combines reconstruction + sparsity penalty
4. **Backpropagation**: Updates weights using gradients
5. **Iteration**: Repeat until convergence

---

## 📈 Applications

- **Feature Learning**: Extracts interpretable features from raw data
- **Image Denoising**: Filters out noise while preserving structure
- **Anomaly Detection**: Sparse activations highlight unusual patterns
- **Pretraining**: Used to initialize deep networks with meaningful weights

---

## 🧠 Intuition Behind Sparsity

- Forces the network to **specialize** neurons for specific features
- Prevents **trivial identity mapping**
- Mimics biological neural systems where only a few neurons fire at once

--- 

## Additional Learning 

- [Deep Learning Tutorial - Sparse Autoencoder By Chris McCormick (2014)](http://mccormickml.com/2014/05/30/deep-learning-tutorial-sparse-autoencoder/)
- [Deep Learning - Sparse AutoEncoders By Eric Wilkinson (2014)](http://www.ericlwilkinson.com/blog/2014/11/19/deep-learning-sparse-autoencoders)
- [K-Sparse AutoEncoders By Alireza Makhzani et al. (2014)](https://arxiv.org/pdf/1312.5663.pdf)
---

## 📚 Recommended Reading

- [Sparse Autoencoders in Deep Learning – GeeksforGeeks](https://www.geeksforgeeks.org/deep-learning/sparse-autoencoders-in-deep-learning/)
- [Stanford Lecture Notes by Andrew Ng](https://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf)
- [Intuitive Explanation of Sparse Autoencoders](https://adamkarvonen.github.io/machine_learning/2024/06/11/sae-intuitions.html)

---

## 🔧 What Is a Denoising Autoencoder?

A **Denoising Autoencoder** is a neural network trained to:
- **Take a noisy or corrupted version of input**
- **Reconstruct the original clean input**

This forces the model to learn **robust features** that capture the underlying structure of the data rather than memorizing it.

---

## 🧱 Architecture Overview

- **Input Layer**: Receives corrupted data (e.g., image with noise)
- **Encoder**: Maps noisy input to a latent representation
- **Decoder**: Reconstructs the original clean input from the latent space
- **Output Layer**: Matches the original (uncorrupted) input

> 🧠 Unlike standard autoencoders, the loss is computed between the output and the **clean input**, not the noisy one.

---

## 🧮 Objective Function

The goal is to minimize the **reconstruction loss**:

$$
L = \|x - \hat{x}\|^2
$$

Where:
- \( x \): Original clean input
- ( $\hat{x}$ ): Reconstructed output from noisy input

---

## 🌪️ Types of Noise Used

| Noise Type         | Description                                      |
|--------------------|--------------------------------------------------|
| **Gaussian Noise** | Add random values from a normal distribution     |
| **Salt & Pepper**  | Randomly flip pixel values to black or white     |
| **Masking Noise**  | Randomly set some input values to zero           |
| **Block Masking**  | Hide entire regions of the input (e.g., image)   |
| **Blurring**       | Apply Gaussian blur to smooth the input          |

These techniques simulate real-world corruption and help the model generalize better.

---

## 🧪 PyTorch Implementation (Simplified)

```python
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_noisy = x + torch.randn_like(x) * 0.2  # Add Gaussian noise
        encoded = self.encoder(x_noisy)
        decoded = self.decoder(encoded)
        return decoded
```

---

## 🎯 Why Use Denoising Autoencoders?

- **Robust Feature Learning**: Learns to ignore noise and focus on structure
- **Improved Generalization**: Avoids overfitting by training on corrupted data
- **Pretraining**: Useful for initializing deep networks
- **Image Restoration**: Effective in removing noise from visual data
- **Anomaly Detection**: Highlights deviations from expected patterns

---

## Additional Learning 

- [Extracting and Composing Robust Features with Denoising AutoEncoders By Pascan Vincet (2008)](http://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf)

---

## 📚 Key Research & Resources

- [GeeksforGeeks: Denoising AutoEncoders in Machine Learning](https://www.geeksforgeeks.org/machine-learning/denoising-autoencoders-in-machine-learning/)
- [University of Washington Lecture Notes on DAEs](https://courses.cs.washington.edu/courses/cse599i/20au/resources/L17_denoising.pdf)
- [Scaler Topics: Exploring Denoising Autoencoders](https://www.scaler.com/topics/deep-learning/denoising-autoencoder/)
- [François Fleuret’s Deep Learning Notes on DAEs](https://fleuret.org/dlc/materials/dlc-handout-7-3-denoising-autoencoders.pdf)

---

## 🔧 What Is a Contractive Autoencoder?

A **Contractive Autoencoder** is a regularized autoencoder that adds a penalty term to the loss function to **reduce the sensitivity of the encoder** to small changes in the input. This encourages the model to learn **stable and invariant features**.

> 📌 Think of it as teaching the model to “contract” its response to tiny input perturbations—hence the name.

---

## 🧱 Architecture Overview

- **Input Layer**: Raw data (e.g., images, vectors)
- **Encoder**: Maps input to latent space
- **Decoder**: Reconstructs input from latent space
- **Output Layer**: Matches original input

The key difference lies in the **loss function**, which includes a **contractive penalty**.

---

## 🧮 Objective Function

The total loss combines:

1. **Reconstruction Loss**  
   $$L_{\text{recon}} = \|x - \hat{x}\|^2$$  
   Measures how well the output matches the input.

2. **Contractive Penalty**  
   $$L_{\text{contractive}} = \lambda \cdot \|J_h(x)\|_F^2$$  
   - \( J_h(x) \): Jacobian of the encoder’s hidden representation w.r.t. input  
   - \( \| \cdot \|_F^2 \): Frobenius norm (sum of squared partial derivatives)  
   - λ: Regularization strength

3. **Total Loss**  
   $$L = L_{\text{recon}} + L_{\text{contractive}}$$

This penalty discourages the encoder from being too sensitive to input changes.

---

## 🧠 Intuition Behind the Jacobian Penalty

- The **Jacobian matrix** captures how much each hidden unit changes with respect to each input feature.
- Penalizing its norm ensures that **small input changes** don’t cause **large changes** in the hidden representation.
- This leads to **smooth, stable, and invariant features**.

---

## 🧪 PyTorch Implementation (Simplified)

```python
def contractive_loss(x, x_hat, hidden, W, lam):
    mse = F.mse_loss(x_hat, x)
    dh = hidden * (1 - hidden)  # derivative of sigmoid
    W_squared = W.pow(2)
    contractive_term = lam * torch.sum(W_squared * dh.pow(2))
    return mse + contractive_term
```

This assumes:
- `hidden` is the encoder output
- `W` is the weight matrix of the encoder
- `lam` is the regularization coefficient

---

## 🎯 Why Use Contractive Autoencoders?

| Benefit                     | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| **Robustness to Noise**     | Learns features that ignore small perturbations                            |
| **Improved Generalization** | Avoids overfitting by focusing on stable patterns                          |
| **Feature Invariance**      | Captures essential structure, ignoring irrelevant transformations          |
| **Smooth Latent Space**     | Useful for downstream tasks like classification or clustering              |

---

## 🔄 Comparison with Other Autoencoders

| Type                  | Key Idea                                      | Robustness Mechanism                     |
|-----------------------|-----------------------------------------------|------------------------------------------|
| **Standard AE**       | Reconstruct input                             | No explicit robustness                   |
| **Denoising AE**      | Reconstruct clean input from noisy version    | Learns to ignore finite perturbations    |
| **Contractive AE**    | Penalize sensitivity to input changes         | Learns to ignore infinitesimal changes   |
| **Sparse AE**         | Enforce sparse hidden activations             | Encourages specialization of neurons     |

---

## Additional Learning 

- [Contrastive Auto-Encoders: Explicit Invariance During the Feature Extraction By Salah Rifai et al(2011)](https://icml.cc/2011/papers/455_icmlpaper.pdf)

--- 

## 📚 Recommended Reading

- [Contractive Autoencoders (CAE) Formulation – apxml.com](https://apxml.com/courses/autoencoders-representation-learning/chapter-3-regularized-autoencoders/contractive-autoencoders-formulation)
- [Contractive Autoencoders Explained with Implementation – OpenGenus](https://iq.opengenus.org/contractive-autoencoder/)
- [GeeksforGeeks: Contractive Autoencoder Overview](https://www.geeksforgeeks.org/deep-learning/contractive-autoencoder-cae/)

---

## 🔧 What Is a Stacked Autoencoder?

A **Stacked Autoencoder** is a deep neural network formed by **stacking multiple autoencoders** on top of each other. Each layer learns to encode the output of the previous layer, allowing the network to build **hierarchical feature representations**.

> Think of it as building a tower of abstraction—each level captures increasingly complex patterns.

---

## 🧱 Architecture Overview

- **Layer 1**: Autoencoder trained on raw input
- **Layer 2**: Autoencoder trained on Layer 1’s encoded output
- **Layer n**: Trained on encoded output of Layer n−1

After training each layer individually, the entire stack is **fine-tuned** using backpropagation.

### Example Structure:
```
Input → AE1 → AE2 → AE3 → Decoder3 → Decoder2 → Decoder1 → Output
```

---

## 🧪 Training Strategy

### 🔹 Phase 1: Layer-wise Pretraining
- Train each autoencoder independently
- Use unsupervised learning (e.g., MSE loss)
- Freeze weights after training each layer

### 🔹 Phase 2: Fine-tuning
- Stack all encoders and decoders
- Train the full network end-to-end using backpropagation
- Optionally use labeled data for supervised fine-tuning

---

## 🧮 Objective Function

Each autoencoder minimizes its own **reconstruction loss**:

$$
L = \|x - \hat{x}\|^2
$$

During fine-tuning, the loss is computed between the final output and the original input.

---

## 🎯 Why Use Stacked Autoencoders?

| Benefit                     | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| **Deep Feature Learning**   | Captures complex, hierarchical patterns                                     |
| **Unsupervised Pretraining**| Useful when labeled data is scarce                                          |
| **Dimensionality Reduction**| Learns compact representations for visualization or clustering              |
| **Transfer Learning**       | Pretrained layers can be reused across tasks                                |
| **Improved Initialization** | Helps avoid poor local minima in deep networks                              |

---

## 🧠 Intuition Behind Layer Stacking

- Each layer learns **increasingly abstract features**
- Lower layers capture edges or textures (in images)
- Higher layers capture shapes, objects, or semantic meaning

---

## 🧪 PyTorch Skeleton (Simplified)

```python
class AE(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Stack multiple AEs
ae1 = AE(784, 512)
ae2 = AE(512, 256)
ae3 = AE(256, 128)
```

Each autoencoder is trained separately, then stacked and fine-tuned.

---

## Additional Learning 

- [Stacked Denoising AutoEncoders Learning Useful Representations in a Deep Network with Local Denoising Criterion By Pascal Vince et al (2010)](http://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf)

---

## 📚 Recommended Reading

- [Autoencoders – IIT Kharagpur Lecture Notes](https://cse.iitkgp.ac.in/~sudeshna/courses/DL17/Autoencoder-15-Mar-17.pdf)
- [Stacked Autoencoders – Towards Data Science](https://towardsdatascience.com/stacked-autoencoders-f0a4391ae282)
- [CSCE Lecture on Autoencoders – University of Nebraska](https://cse.unl.edu/~sscott/teach/Classes/cse496S19/slides/05-Autoencoders.pdf)

---

## 🔧 What Is a Deep Autoencoder?

A **Deep Autoencoder** is an autoencoder architecture with **multiple hidden layers** in both the encoder and decoder. This depth allows the network to learn **complex, nonlinear mappings** and extract **high-level features** from raw data.

> Think of it as a multi-stage compression and reconstruction pipeline, where each layer refines the representation further.

---

## 🧱 Architecture Overview

- **Encoder**: Multiple layers that progressively reduce dimensionality
- **Latent Space (Bottleneck)**: The most compressed representation
- **Decoder**: Symmetric layers that reconstruct the input from the latent space

### Example:
```
Input → 1024 → 512 → 256 → 128 → Bottleneck → 128 → 256 → 512 → 1024 → Output
```

> The encoder and decoder are often mirror images, but this is not a strict requirement.

---

## 🧮 Objective Function

The goal is to minimize **reconstruction loss**:

$$
L = \|x - \hat{x}\|^2
$$

Where:
- \( x \): Original input
- \( \hat{x} \): Reconstructed output

You can also use:
- **Binary Cross-Entropy** for binary data
- **KL Divergence** in regularized variants like VAEs

---

## 🧪 PyTorch Skeleton (Simplified)

```python
class DeepAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
```

---

## 🎯 Why Use Deep Autoencoders?

| Benefit                     | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| **Hierarchical Features**   | Learns layered abstractions from raw data                                  |
| **Nonlinear Compression**   | Captures complex patterns beyond linear methods like PCA                   |
| **Robustness**              | Handles noise and variability in input                                     |
| **Transfer Learning**       | Pretrained encoder can be reused for other tasks                           |
| **Dimensionality Reduction**| Useful for visualization and clustering                                    |

---

## 🔄 Comparison with Shallow Autoencoders

| Feature                  | Shallow AE           | Deep AE                        |
|--------------------------|----------------------|--------------------------------|
| Layers                   | 1–2 hidden layers     | 3+ hidden layers               |
| Feature Complexity       | Low-level             | High-level, abstract           |
| Generalization           | Limited               | Better with regularization     |
| Training Time            | Fast                  | Slower, needs careful tuning   |

---

## 🧠 Training Tips

- Use **layer-wise pretraining** to initialize weights
- Apply **dropout** and **batch normalization** to prevent overfitting
- Choose appropriate **activation functions** (ReLU, Sigmoid, etc.)
- Monitor **reconstruction loss** and **latent space behavior**

--- 

## Additional Learning 

- [Reducing the Dimentionality of Data with Neural Networks By Geoffery Hinton et al (2006)](https://www.cs.toronto.edu/~hinton/science.pdf)

---

## 📚 Recommended Reading

- [Deep Autoencoders Tutorial – PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/08-deep-autoencoders.html)
- [Autoencoders & Generative Models Notes – Studocu](https://www.studocu.com/in/document/panimalar-engineering-college/deep-learning/unit5-autoencoder-notes/114394619)
- [Autoencoders in Machine Learning – GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/auto-encoders/)

---
