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
