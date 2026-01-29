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
