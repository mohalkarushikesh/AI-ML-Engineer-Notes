# Deep Learning Architectures — In-Depth, with Pseudocode

> Scope: the major neural-network architectures, the math that defines each, clean language-neutral pseudocode, and the "what / why / when" for each. Pseudocode is NumPy-ish: `@` = matrix multiply, `*` = elementwise, shapes noted where they matter. Concepts here are stable fundamentals.

**Notation:** `x` input, `W`/`b` weights/bias, `a` activation, `z` pre-activation, `σ` sigmoid, `⊙` elementwise product (written `*`), `d_k` key dimension, `N(0,I)` standard normal.

---

## 0. The common skeleton

Every architecture below is: **parameters → forward pass → loss → gradients (backprop) → optimizer step**, repeated.

```
for epoch in range(E):
    for batch (x, y) in data:
        y_hat = model.forward(x)          # forward
        loss  = loss_fn(y_hat, y)         # scalar
        grads = backward(loss)            # autodiff / backprop
        params = optimizer.step(params, grads)
```

Autodiff frameworks (PyTorch/JAX/TF) compute `backward` for you via the chain rule; the pseudocode below shows the forward math and, where instructive, the manual gradients.

---

## 1. Perceptron & Multi-Layer Perceptron (MLP)

**What:** stacked fully-connected layers with nonlinearities. The universal function approximator; baseline for tabular data.

**Single neuron:** `a = activation(w · x + b)`

### Forward (L layers)
```
def mlp_forward(x, params):
    a = x
    cache = [a]
    for l in 1..L:
        z = W[l] @ a + b[l]
        a = activation(z)          # ReLU for hidden, softmax/linear/sigmoid for output
        cache.append((z, a))
    return a, cache
```

### Backprop (chain rule, one sample; batch = add a dim)
```
def mlp_backward(y_hat, y, cache, params):
    dz = dLoss_dz_output(y_hat, y)              # e.g. (softmax - onehot) for CE
    for l in L..1:
        dW[l] = dz @ a[l-1].T
        db[l] = sum_over_batch(dz)
        if l > 1:
            da = W[l].T @ dz
            dz = da * activation_deriv(z[l-1])  # gate the gradient by the local slope
    return grads
```

**Key idea:** backprop = reverse-mode chain rule; each layer multiplies the incoming gradient by its local Jacobian. Vanishing/exploding gradients arise when these local slopes are consistently <1 or >1 across many layers.

---

## 2. Activation & loss functions (building blocks)

**Activations**
| Fn | Formula | Use |
|---|---|---|
| ReLU | `max(0, z)` | Hidden default; cheap, sparse, no vanishing for z>0 |
| Leaky/GELU/SiLU | `max(αz,z)` / smooth variants | Fix dead ReLUs; GELU standard in Transformers |
| Sigmoid | `1/(1+e^-z)` | Binary output / gates |
| Tanh | `(e^z−e^−z)/(e^z+e^−z)` | Zero-centered; RNN states |
| Softmax | `e^{z_i}/Σe^{z_j}` | Multiclass output (probabilities) |

**Losses**
| Task | Loss |
|---|---|
| Regression | MSE `mean((y−ŷ)²)`, MAE, Huber |
| Binary class | BCE `−[y log ŷ + (1−y) log(1−ŷ)]` |
| Multiclass | Cross-entropy `−Σ y_i log ŷ_i` |
| Embeddings/metric | Contrastive, Triplet, InfoNCE |

---

## 3. Convolutional Neural Networks (CNNs)

**What:** weight-shared local filters → translation-equivariant feature detectors. **When:** images, audio spectrograms, any grid data. **Why:** parameter sharing + locality → far fewer params than MLP, built-in spatial priors.

### Convolution
```
def conv2d(x, kernels, bias, stride=1, pad=0):
    # x: [H,W,C_in], kernels: [kh,kw,C_in,C_out]
    x = zero_pad(x, pad)
    for oc in range(C_out):
      for i in range(0, out_H):
        for j in range(0, out_W):
            patch = x[i*stride : i*stride+kh, j*stride : j*stride+kw, :]
            out[i,j,oc] = sum(patch * kernels[:,:,:,oc]) + bias[oc]
    return out      # [out_H, out_W, C_out]
# out_size = floor((H + 2*pad - kh)/stride) + 1
```

### Pooling (downsample, add invariance)
```
max_pool[i,j] = max(x[i*s:i*s+k, j*s:j*s+k])   # or mean for avg-pool
```

### Classic block
```
block(x) = Pool(Activation(BatchNorm(Conv(x))))
```

### Residual block (ResNet) — enables very deep nets
```
def residual_block(x):
    h = Conv(x); h = BN(h); h = ReLU(h)
    h = Conv(h); h = BN(h)
    return ReLU(h + x)          # skip connection: gradient highway
# If channel/spatial dims change, project x with a 1x1 conv.
```

**Why skips work:** the identity path lets gradients flow unimpeded, so `∂loss/∂x` always has a `+1` term → mitigates vanishing gradients, trains 100+ layers.

**Notable CNN families:** VGG (simple stacks), Inception (multi-scale parallel convs), **ResNet** (skips), DenseNet (concat skips), MobileNet (**depthwise-separable** convs for efficiency), EfficientNet (compound scaling), U-Net (encoder-decoder + skip concats, for segmentation).

```
# Depthwise-separable conv (MobileNet) = cheap convolution
sep_conv(x) = pointwise_1x1( depthwise_per_channel(x) )
```

---

## 4. Recurrent networks (RNN, LSTM, GRU)

**What:** process sequences by carrying a hidden state across time. **When:** text, time series, audio (pre-Transformer). **Why/limits:** share weights across time, but vanilla RNNs suffer vanishing gradients over long ranges.

### Vanilla RNN cell
```
def rnn_cell(x_t, h_prev):
    h_t = tanh(W_xh @ x_t + W_hh @ h_prev + b)
    return h_t
# Unroll over t; train with Backprop Through Time (BPTT).
```

### LSTM cell (gates solve long-range memory)
```
def lstm_cell(x_t, h_prev, c_prev):
    z = concat(h_prev, x_t)
    f_t  = sigmoid(W_f @ z + b_f)      # forget: what to drop from cell
    i_t  = sigmoid(W_i @ z + b_i)      # input:  what to write
    c~_t = tanh   (W_c @ z + b_c)      # candidate values
    c_t  = f_t * c_prev + i_t * c~_t   # updated cell state (the memory highway)
    o_t  = sigmoid(W_o @ z + b_o)      # output gate
    h_t  = o_t * tanh(c_t)
    return h_t, c_t
```
**Why it works:** the cell state `c_t` has an additive update → gradient flows without repeated multiplication (like a residual connection through time).

### GRU cell (lighter, often comparable)
```
def gru_cell(x_t, h_prev):
    z_t  = sigmoid(W_z @ concat(h_prev, x_t))      # update gate
    r_t  = sigmoid(W_r @ concat(h_prev, x_t))      # reset gate
    h~_t = tanh(W_h @ concat(r_t * h_prev, x_t))   # candidate
    h_t  = (1 - z_t) * h_prev + z_t * h~_t
    return h_t
```

### Encoder–decoder (Seq2Seq) + attention
```
# Encoder produces hidden states H = [h_1 ... h_n]
# Decoder at step t, with its state s_t:
scores  = [score(s_t, h_i) for h_i in H]      # Bahdanau: vᵀ tanh(W1 s + W2 h)
weights = softmax(scores)
context = sum(weights[i] * h_i for i)          # weighted focus over the input
out_t   = decode(s_t, context)
```
Attention removed the single-vector bottleneck of Seq2Seq — the direct ancestor of the Transformer.

---

## 5. Transformers (the modern backbone)

**What:** sequence model built entirely on **attention** — no recurrence, fully parallel over positions. **When:** NLP, vision (ViT), audio, multimodal, and all LLMs. **Why:** captures long-range dependencies in O(1) path length and parallelizes across the sequence.

### Scaled dot-product attention
```
def attention(Q, K, V, mask=None):
    # Q:[n,d_k] K:[m,d_k] V:[m,d_v]
    scores = (Q @ K.T) / sqrt(d_k)       # [n,m] similarity
    if mask: scores += mask              # -inf where disallowed (causal/padding)
    A = softmax(scores, axis=-1)
    return A @ V                         # [n,d_v] weighted values
```
The `/sqrt(d_k)` keeps dot products from growing with dimension (which would saturate softmax).

### Multi-head attention (look at multiple subspaces)
```
def multi_head_attention(X, h):          # X:[n, d_model]
    for i in 1..h:
        Q_i = X @ W_Q[i];  K_i = X @ W_K[i];  V_i = X @ W_V[i]   # project to d_k
        head[i] = attention(Q_i, K_i, V_i, mask)
    return concat(head[1..h]) @ W_O       # mix heads back to d_model
```

### Positional encoding (inject order, since attention is permutation-invariant)
```
PE[pos, 2i]   = sin(pos / 10000^(2i/d_model))
PE[pos, 2i+1] = cos(pos / 10000^(2i/d_model))
X = embedding(tokens) + PE               # (or learned / rotary (RoPE) positions)
```

### Encoder block (pre-norm variant, common today)
```
def encoder_block(x):
    x = x + multi_head_attention(layer_norm(x))   # self-attention + residual
    x = x + ffn(layer_norm(x))                     # position-wise FFN + residual
    return x
# ffn(x) = Linear(d_ff) -> GELU -> Linear(d_model),  d_ff ≈ 4*d_model
```

### Decoder block (adds masking + cross-attention)
```
def decoder_block(x, enc_out):
    x = x + masked_self_attention(layer_norm(x))       # causal mask (no peeking ahead)
    x = x + cross_attention(layer_norm(x), enc_out)    # attend to encoder (for seq2seq)
    x = x + ffn(layer_norm(x))
    return x
```

### Three families
| Family | Blocks | Objective | Examples / use |
|---|---|---|---|
| **Encoder-only** | bidirectional self-attn | Masked LM | BERT — understanding, embeddings, classification |
| **Decoder-only** | causal self-attn | Next-token prediction | GPT/Llama/Claude-style LLMs — generation |
| **Encoder-decoder** | both | Seq2seq | T5, translation, summarization |
| **Vision (ViT)** | encoder on image **patches** | classification | split image into patches → tokens → Transformer |

### Minimal decoder-only LM generation loop
```
tokens = prompt
while not done:
    logits = transformer(tokens)[-1]      # next-token distribution
    next   = sample(softmax(logits / T))  # T=temperature; top-k/top-p to truncate
    tokens.append(next)
```

---

## 6. Autoencoders & representation learning

**What:** learn compressed latent codes by reconstructing the input. **When:** denoising, anomaly detection, dimensionality reduction, pretraining.

```
z     = encoder(x)        # bottleneck latent (dim << input)
x_hat = decoder(z)
loss  = reconstruction_loss(x, x_hat)      # MSE / BCE
# Variants: denoising AE (corrupt input), sparse AE, masked AE (MAE)
```

---

## 7. Variational Autoencoder (VAE)

**What:** probabilistic autoencoder that learns a *distribution* over latents → can **generate** by sampling. **Why the trick:** sampling isn't differentiable, so use the **reparameterization trick**.

```
def vae_step(x):
    mu, logvar = encoder(x)                       # latent Gaussian params
    eps = sample N(0, I)
    z   = mu + exp(0.5*logvar) * eps              # reparameterization (differentiable)
    x_hat = decoder(z)

    recon = reconstruction_loss(x, x_hat)
    kl    = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))   # KL(q(z|x) || N(0,I))
    return recon + beta * kl                      # ELBO (beta-VAE weights the KL)
```
The KL term regularizes the latent space toward a standard normal so you can sample `z ~ N(0,I)` and decode to new samples.

---

## 8. Generative Adversarial Networks (GAN)

**What:** two nets in a minimax game — Generator fakes data, Discriminator tells real from fake. **When:** sharp image synthesis (historically), super-resolution, style transfer.

```
# D wants to classify correctly; G wants to fool D.
for step:
    # --- train Discriminator ---
    z = sample N(0,I); fake = G(z)
    loss_D = -(log D(real) + log(1 - D(fake)))    # maximize log D(real)+log(1-D(fake))
    update(D, loss_D)

    # --- train Generator ---
    z = sample N(0,I); fake = G(z)
    loss_G = -log D(fake)                          # non-saturating: maximize log D(fake)
    update(G, loss_G)
```
**Failure modes:** mode collapse (G makes few distinct samples), training instability. Fixes: Wasserstein loss (WGAN-GP), spectral norm, two-timescale LR (TTUR).

---

## 9. Diffusion models (DDPM) — the modern image/video generators

**What:** learn to *denoise*. Forward process gradually adds Gaussian noise; a network learns the reverse. **When:** state-of-the-art image/audio/video generation (Stable Diffusion, etc.).

```
# Forward (fixed): x_t = sqrt(ᾱ_t)*x_0 + sqrt(1-ᾱ_t)*noise,  noise ~ N(0,I)
# ᾱ_t decreases with t → by t=T, x_T is ~pure noise.

def diffusion_train_step(x_0):
    t      = random timestep in [1, T]
    noise  = sample N(0, I)
    x_t    = sqrt(alpha_bar[t])*x_0 + sqrt(1 - alpha_bar[t])*noise
    pred   = eps_theta(x_t, t)              # network predicts the added noise
    return mse(noise, pred)                 # simple denoising objective

def sample():                               # reverse process
    x = sample N(0, I)                      # start from pure noise
    for t in T..1:
        eps = eps_theta(x, t)
        x   = denoise_step(x, eps, t)       # subtract predicted noise, add a little back
    return x                                # a fresh sample
```
Conditioning (text→image) adds a prompt embedding via cross-attention; **classifier-free guidance** steers samples toward the condition. The denoiser is usually a **U-Net** (or a Transformer, "DiT").

---

## 10. Graph Neural Networks (GNN) — message passing

**What:** learn on graphs (molecules, social nets, knowledge graphs) via neighbor aggregation. **When:** relational/irregular data.

```
# One message-passing layer, for each node v:
def gnn_layer(h):
    for v in nodes:
        msgs   = [message(h[u], edge(u,v)) for u in neighbors(v)]
        agg    = aggregate(msgs)            # sum / mean / max  (permutation-invariant)
        h_new[v] = update(h[v], agg)        # e.g. MLP or GRU
    return h_new
# GCN closed form: H' = σ( D^{-1/2} (A+I) D^{-1/2} H W )   (normalized neighbor averaging)
# GAT: weight neighbors with attention. GraphSAGE: sample + aggregate for scale.
```

---

## 11. Core training machinery (applies to all)

### Optimizers
```
# SGD + momentum
v = μ*v - lr*g;   θ = θ + v

# Adam (adaptive, the default)
m = β1*m + (1-β1)*g                 # 1st moment (mean)
s = β2*s + (1-β2)*g^2               # 2nd moment (variance)
m̂ = m/(1-β1^t);  ŝ = s/(1-β2^t)    # bias correction
θ = θ - lr * m̂ / (sqrt(ŝ) + ε)
# AdamW = Adam with decoupled weight decay (standard for Transformers)
```

### Normalization (stabilize/accelerate training)
| Type | Normalizes over | Use |
|---|---|---|
| **BatchNorm** | the batch (per channel) | CNNs; depends on batch stats |
| **LayerNorm** | the features (per sample) | Transformers/RNNs; batch-independent |
| **GroupNorm / RMSNorm** | channel groups / RMS only | small batches; LLMs (RMSNorm) |

### Regularization & stability
- **Dropout** — randomly zero activations (p≈0.1–0.5) → prevents co-adaptation.
- **Weight decay (L2)** — shrink weights.
- **Early stopping** — halt when val loss stops improving.
- **Data augmentation** — flips/crops/mixup/cutout (vision), token masking (NLP).
- **Gradient clipping** — cap `‖g‖` to tame exploding gradients (RNNs/Transformers).
- **Residual connections + normalization** — the structural fixes for deep-net gradient flow.

### Initialization
- **He/Kaiming** for ReLU nets; **Xavier/Glorot** for tanh/sigmoid. Bad init → vanishing/exploding signals from layer 1.

### Learning-rate schedules
- Warmup → cosine decay (Transformers); step decay; one-cycle. LR is the single most impactful hyperparameter.

### Transfer learning / fine-tuning
```
model = pretrained_backbone()        # trained on large corpus
freeze(model.layers[:-k])            # keep general features
replace(model.head, new_task_head)
train(model, small_task_dataset)     # adapt to your task
# Parameter-efficient variants: LoRA (low-rank adapters), adapters, prefix-tuning.
```

---

## 12. Choosing an architecture

| Data / task | Start with |
|---|---|
| Tabular | Gradient-boosted trees first; MLP if deep features help |
| Images (classify/detect/segment) | CNN (ResNet/EfficientNet) or **ViT**; U-Net for segmentation |
| Sequences / time series | Transformer; LSTM/GRU for small data or streaming |
| Text (understand) | Encoder Transformer (BERT-style) |
| Text (generate) / chat | Decoder-only Transformer (GPT/Llama-style LLM) |
| Translation / summarization | Encoder-decoder Transformer (T5-style) |
| Generate images/audio/video | **Diffusion** (U-Net/DiT); GANs for some real-time cases |
| Learn compact codes / anomalies | Autoencoder / VAE |
| Graphs / relational | GNN (GCN/GAT/GraphSAGE) |

---

### Highest-yield takeaways
1. **Backprop = reverse-mode chain rule**; each layer gates the gradient by its local slope → deep nets need residuals + normalization + good init to flow gradients.
2. **CNNs** = weight sharing + locality; **residual skips** are what make them deep.
3. **LSTM/GRU** solve RNN vanishing gradients via **additive, gated** state updates.
4. **Attention** = `softmax(QKᵀ/√d_k)V`; **multi-head** reads multiple subspaces; **positional encodings** add order.
5. **Transformer families:** encoder-only (BERT/understand), decoder-only (GPT/generate), encoder-decoder (T5/seq2seq); **ViT** = Transformer on image patches.
6. **VAE** uses the **reparameterization trick** + KL; **GAN** is a minimax game; **diffusion** learns to denoise and now leads image generation.
7. **AdamW + warmup/cosine + LayerNorm + dropout + grad-clipping** is the default modern training recipe.
8. **Transfer learning / LoRA** beats training from scratch for most real tasks.
