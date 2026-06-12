# Deep Learning Cheatsheet

---

## Core Concepts

### The Neuron (Perceptron)
```
Inputs → Weighted Sum → Activation Function → Output
  x₁
  x₂  →  z = Σ(wᵢxᵢ) + b  →  a = f(z)  →  ŷ
  x₃
```

### Key Terms
| Term | Definition |
|------|------------|
| **Parameter** | Learnable weight or bias |
| **Hyperparameter** | Manually set value (lr, layers, batch size) |
| **Epoch** | One full pass over the training dataset |
| **Batch size** | Samples processed before a weight update |
| **Iteration** | One forward + backward pass on a batch |
| **Overfitting** | Model memorizes training data, fails to generalize |
| **Underfitting** | Model too simple to capture patterns |

---

## Neural Network Anatomy

```
Input Layer → Hidden Layers → Output Layer
   (X)       (representations)    (ŷ)
```

### Layer Types
| Layer | Use Case |
|-------|----------|
| **Dense / Fully Connected** | General purpose, classification, regression |
| **Convolutional (Conv2D)** | Images, spatial data |
| **Recurrent (RNN/LSTM/GRU)** | Sequences, time series, text |
| **Transformer / Attention** | NLP, vision, multimodal |
| **Embedding** | Categorical inputs, word vectors |
| **Batch Norm** | Stabilize & accelerate training |
| **Dropout** | Regularization, prevent overfitting |
| **Pooling** | Downsample spatial dimensions (CNN) |

---

## Activation Functions

| Function | Formula | Range | Use When |
|----------|---------|-------|----------|
| **ReLU** | max(0, x) | [0, ∞) | Default for hidden layers |
| **Leaky ReLU** | max(αx, x) | (-∞, ∞) | Dying ReLU problem |
| **GELU** | x·Φ(x) | (-∞, ∞) | Transformers (BERT, GPT) |
| **Sigmoid** | 1/(1+e⁻ˣ) | (0, 1) | Binary output |
| **Softmax** | eˣⁱ/Σeˣʲ | (0, 1), sums to 1 | Multi-class output |
| **Tanh** | (eˣ−e⁻ˣ)/(eˣ+e⁻ˣ) | (-1, 1) | RNNs, zero-centered |
| **Swish** | x·sigmoid(x) | (-∞, ∞) | EfficientNet, modern CNNs |

> ⚠️ **Dying ReLU:** neurons stuck at 0 gradient — use Leaky ReLU or initialize carefully.

---

## Loss Functions

### Regression
| Loss | Formula | Notes |
|------|---------|-------|
| **MSE** | mean((y − ŷ)²) | Penalizes outliers heavily |
| **MAE** | mean(\|y − ŷ\|) | Robust to outliers |
| **Huber** | combo of MSE+MAE | Best of both worlds |

### Classification
| Loss | Use When |
|------|----------|
| **Binary Cross-Entropy** | Binary classification (sigmoid output) |
| **Categorical Cross-Entropy** | Multi-class (softmax output, one-hot labels) |
| **Sparse Categorical CE** | Multi-class (integer labels) |
| **Focal Loss** | Class imbalance (detection tasks) |
| **KL Divergence** | Distribution matching (VAEs) |

---

## Optimizers

| Optimizer | Notes |
|-----------|-------|
| **SGD** | Simple, requires tuning momentum & lr |
| **SGD + Momentum** | Faster convergence, less oscillation |
| **RMSProp** | Adaptive lr, good for RNNs |
| **Adam** | Adaptive lr + momentum, most popular default |
| **AdamW** | Adam + decoupled weight decay (transformers) |
| **Lion** | Memory-efficient, emerging alternative |

### Learning Rate Rules of Thumb
```
Too high  → loss diverges / oscillates
Too low   → very slow convergence
Typical range: 1e-4 to 1e-2
Transformers: 1e-5 to 3e-4 with warmup
```

### LR Schedules
| Schedule | Description |
|----------|-------------|
| **Step decay** | Multiply by factor every N epochs |
| **Cosine annealing** | Smooth decay following cosine curve |
| **Warmup + decay** | Ramp up then decay (standard for transformers) |
| **ReduceLROnPlateau** | Reduce when metric stops improving |
| **Cyclic LR** | Oscillate between min/max lr |

---

## Backpropagation

```
Forward pass:   X → [layers] → ŷ → Loss
Backward pass:  ∂Loss/∂W via chain rule → update weights

W = W - lr × ∂Loss/∂W
```

### Gradient Issues
| Problem | Symptom | Fix |
|---------|---------|-----|
| **Vanishing gradients** | Deep nets stop learning | ReLU, skip connections, BatchNorm |
| **Exploding gradients** | Loss goes NaN | Gradient clipping, lower lr |
| **Saddle points** | Training stalls | Momentum-based optimizers |

---

## Regularization Techniques

| Technique | How It Works |
|-----------|-------------|
| **L1 (Lasso)** | Adds Σ\|w\| to loss — promotes sparsity |
| **L2 (Ridge / Weight Decay)** | Adds Σw² to loss — shrinks weights |
| **Dropout** | Randomly zero neurons during training (p=0.1–0.5) |
| **BatchNorm** | Normalize layer inputs — acts as implicit regularizer |
| **Data augmentation** | Artificially expand training data |
| **Early stopping** | Stop when val loss stops improving |
| **Label smoothing** | Soften one-hot targets (e.g. 0.9 / 0.1) |
| **Mixup / CutMix** | Interpolate training samples and labels |

---

## Convolutional Neural Networks (CNN)

```
Input Image → [Conv → BN → ReLU → Pool] × N → Flatten → Dense → Output
```

### Key Parameters
| Parameter | Description |
|-----------|-------------|
| **Filters (kernels)** | Number of feature detectors |
| **Kernel size** | Spatial extent (3×3, 5×5) |
| **Stride** | Step size for sliding the kernel |
| **Padding** | `same` keeps dims; `valid` reduces dims |
| **Receptive field** | Input area influencing one output neuron |

### Output Size Formula
```
Output = floor((Input + 2×Padding − Kernel) / Stride) + 1
```

### Landmark CNN Architectures
| Model | Year | Innovation |
|-------|------|-----------|
| LeNet-5 | 1998 | First practical CNN |
| AlexNet | 2012 | Deep learning renaissance |
| VGG-16/19 | 2014 | Depth with small 3×3 filters |
| ResNet | 2015 | Skip connections, 152 layers |
| EfficientNet | 2019 | Compound scaling |
| ConvNeXt | 2022 | CNN redesigned like a Transformer |

---

## Recurrent Neural Networks (RNN)

```
hₜ = tanh(Wₕhₜ₋₁ + Wₓxₜ + b)
```

### RNN Variants
| Model | Solves | Notes |
|-------|--------|-------|
| **Vanilla RNN** | — | Vanishing gradient problem |
| **LSTM** | Long-term dependencies | Gates: input, forget, output |
| **GRU** | Long-term dependencies | Simpler than LSTM, often equal |
| **Bidirectional** | Context from both directions | Common in NLP |

### LSTM Gates
```
Forget gate:  fₜ = σ(Wf·[hₜ₋₁, xₜ] + bf)
Input gate:   iₜ = σ(Wi·[hₜ₋₁, xₜ] + bi)
Output gate:  oₜ = σ(Wo·[hₜ₋₁, xₜ] + bo)
Cell state:   Cₜ = fₜ⊙Cₜ₋₁ + iₜ⊙tanh(Wc·[hₜ₋₁, xₜ])
```

---

## Transformers & Attention

### Scaled Dot-Product Attention
```
Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V
```

### Multi-Head Attention
```
MultiHead = Concat(head₁, ..., headₕ) · Wᴼ
where headᵢ = Attention(QWᵢQ, KWᵢK, VWᵢV)
```

### Transformer Block
```
Input → LayerNorm → Multi-Head Attention → Residual
      → LayerNorm → Feed-Forward Network  → Residual → Output
```

### Key Architectures
| Model | Type | Use |
|-------|------|-----|
| **BERT** | Encoder-only | Classification, embeddings |
| **GPT** | Decoder-only | Text generation |
| **T5 / BART** | Encoder-Decoder | Seq2seq, translation, summarization |
| **ViT** | Encoder (image patches) | Image classification |
| **CLIP** | Dual encoder | Vision-language |

---

## Normalization Layers

| Type | Normalizes Over | Best For |
|------|----------------|----------|
| **Batch Norm** | Batch dimension | CNNs, large batches |
| **Layer Norm** | Feature dimension | Transformers, RNNs |
| **Group Norm** | Groups of channels | Small batch / detection |
| **Instance Norm** | Per sample, per channel | Style transfer |

---

## Weight Initialization

| Method | Use With |
|--------|---------|
| **Xavier / Glorot** | Sigmoid, Tanh activations |
| **He / Kaiming** | ReLU and variants |
| **Orthogonal** | RNNs |
| **Zero init** | Biases only — never weights |

> ⚠️ Never initialize all weights to zero — symmetry breaking fails.

---

## Training Checklist

```
□ Normalize inputs (zero mean, unit variance)
□ Start with a small model, verify it can overfit 1 batch
□ Use Adam + lr=1e-3 as a baseline
□ Monitor train vs. val loss (divergence = overfitting)
□ Use learning rate finder if unsure
□ Clip gradients if loss is unstable (max_norm=1.0)
□ Log metrics with TensorBoard / W&B
□ Save best model checkpoint by val metric
□ Evaluate on a held-out test set only once
```

---

## Transfer Learning

```
Pretrained Model (ImageNet / large corpus)
        │
        ▼
  Freeze base layers
        │
        ▼
  Add task-specific head
        │
        ▼
  Fine-tune on your data
```

### Strategies
| Strategy | When |
|----------|------|
| **Feature extraction** | Small dataset, similar domain |
| **Fine-tune top layers** | Medium dataset |
| **Full fine-tuning** | Large dataset |
| **LoRA / PEFT** | LLMs, parameter-efficient fine-tuning |

---

## Generative Models

| Model | Mechanism | Strengths |
|-------|-----------|-----------|
| **GAN** | Generator vs. Discriminator | Sharp images |
| **VAE** | Encode to latent distribution | Smooth latent space |
| **Diffusion** | Denoise from Gaussian noise | SOTA image quality |
| **Flow** | Invertible transformations | Exact likelihood |
| **Autoregressive** | Predict next token/pixel | Text, audio |

---

## Evaluation Metrics

### Classification
| Metric | Formula | Notes |
|--------|---------|-------|
| **Accuracy** | correct / total | Misleading on imbalanced data |
| **Precision** | TP / (TP+FP) | How many predicted positives are correct |
| **Recall** | TP / (TP+FN) | How many actual positives are caught |
| **F1** | 2·P·R / (P+R) | Balance of precision & recall |
| **AUC-ROC** | Area under ROC curve | Threshold-independent |

### Regression
`MAE, RMSE, R², MAPE`

### Generation
`FID (images), BLEU / ROUGE (text), Perplexity (LMs)`

---

## Hardware & Optimization

| Technique | Benefit |
|-----------|---------|
| **Mixed precision (fp16/bf16)** | 2× memory, faster on modern GPUs |
| **Gradient checkpointing** | Trade compute for memory |
| **Gradient accumulation** | Simulate large batches on small GPUs |
| **torch.compile** | Graph-level optimization (PyTorch 2+) |
| **Flash Attention** | Memory-efficient attention |
| **Data parallelism (DDP)** | Multi-GPU training |
| **Model parallelism** | Very large models across GPUs |

---

## Key Libraries

| Library | Purpose |
|---------|---------|
| **PyTorch** | Research & production (most popular) |
| **TensorFlow / Keras** | Production & deployment |
| **JAX** | Functional, XLA-compiled, Google research |
| **Hugging Face** | Pretrained models, datasets, training |
| **Lightning** | PyTorch training boilerplate |
| **timm** | SOTA vision models |
| **Weights & Biases** | Experiment tracking |
| **ONNX** | Model interoperability & export |

---

## Quick PyTorch Training Loop

```python
model = MyModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model.train()
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_loss = sum(criterion(model(X.to(device)), y.to(device))
                       for X, y in val_loader) / len(val_loader)
    print(f"Epoch {epoch} | Val Loss: {val_loss:.4f}")
```

---

*Last updated: 2025 · Always validate on held-out data — benchmark on your specific task and domain.*
