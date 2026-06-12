# Mathematics Cheat Sheet — AI/ML · Deep Learning · NLP · Computer Vision

---

## Table of Contents

1. [Linear Algebra](#1-linear-algebra)
2. [Calculus & Optimization](#2-calculus--optimization)
3. [Probability & Information Theory](#3-probability--information-theory)
4. [ML Core — Loss Functions & Models](#4-ml-core--loss-functions--models)
5. [Deep Learning](#5-deep-learning)
6. [Transformer Architecture](#6-transformer-architecture)
7. [Generative Models](#7-generative-models)
8. [NLP](#8-nlp)
9. [Computer Vision](#9-computer-vision)

---

## 1. Linear Algebra

### Matrix Operations

```
AB ≠ BA                    (non-commutative)
(AB)ᵀ = BᵀAᵀ
(AB)⁻¹ = B⁻¹A⁻¹
det(AB) = det(A) · det(B)
```

> Shape rule: `(m×k)(k×n) → m×n`. A is invertible iff `det(A) ≠ 0`.

### Eigendecomposition

```
Av = λv
A = QΛQ⁻¹
Q = [v₁ | v₂ | … | vₙ],  Λ = diag(λ₁, …, λₙ)
```

- Symmetric A → `A = QΛQᵀ` (Q orthogonal)
- Powers: `Aᵏ = QΛᵏQᵀ`

### Singular Value Decomposition (SVD)

```
A = UΣVᵀ

U  — left singular vectors  (m×m, orthogonal)
Σ  — singular values        (m×n, diagonal, σᵢ ≥ 0)
V  — right singular vectors (n×n, orthogonal)
```

**Rank-k approximation:** `Aₖ = UₖΣₖVₖᵀ`  
→ Basis of PCA, LSA, recommender systems.

### Norms & Distance

| Norm | Formula |
|------|---------|
| L1 | `‖x‖₁ = Σ|xᵢ|` |
| L2 (Euclidean) | `‖x‖₂ = √(Σxᵢ²)` |
| L∞ | `‖x‖∞ = max|xᵢ|` |
| Frobenius | `‖A‖F = √(Σᵢⱼ aᵢⱼ²)` |

**Cosine similarity:**
```
cos(θ) = (a · b) / (‖a‖ · ‖b‖)
```

---

## 2. Calculus & Optimization

### Chain Rule & Gradient

```
∂/∂x[f(g(x))] = f'(g(x)) · g'(x)

∇f(x) = [∂f/∂x₁, …, ∂f/∂xₙ]ᵀ

∇(Ax)    = Aᵀ
∇(xᵀAx) = (A + Aᵀ)x
```

- **Jacobian:** `J ∈ ℝᵐˣⁿ`, `Jᵢⱼ = ∂fᵢ/∂xⱼ`
- **Hessian:** `H = ∇²f`

### Gradient Descent Family

```
GD:        θ ← θ − η∇L(θ)

Momentum:  v ← βv + ∇L
           θ ← θ − ηv

Adam:      m ← β₁m + (1−β₁)g
           v ← β₂v + (1−β₂)g²
           m̂ = m/(1−β₁ᵗ),  v̂ = v/(1−β₂ᵗ)
           θ ← θ − η · m̂ / (√v̂ + ε)
```

Typical: `η ≈ 1e-3`, `β₁ = 0.9`, `β₂ = 0.999`, `ε = 1e-8`

### Taylor Expansion (Second Order)

```
f(x+δ) ≈ f(x) + δᵀ∇f + ½δᵀHδ
```

Second-order optimality: `∇f(x*) = 0` AND `H ⪰ 0`  
Newton step: `δ = −H⁻¹∇f`

### Lagrangian & KKT Conditions

```
L(x,λ) = f(x) + Σλᵢgᵢ(x)

KKT:  ∇f + Σλᵢ∇gᵢ = 0
      λᵢ ≥ 0
      λᵢgᵢ(x) = 0   (complementary slackness)
```

---

## 3. Probability & Information Theory

### Fundamentals

```
P(A∩B) = P(A|B)P(B) = P(B|A)P(A)

Bayes:   P(A|B) = P(B|A)P(A) / P(B)

E[X] = Σ xᵢpᵢ  or  ∫ x f(x) dx

Var(X) = E[X²] − (E[X])²
```

### Key Distributions

| Distribution | PDF / PMF |
|---|---|
| Gaussian | `(2πσ²)^(-½) exp(−(x−μ)²/2σ²)` |
| Bernoulli | `pˣ(1−p)¹⁻ˣ` |
| Categorical | `∏ pₖˣᵏ` |
| Dirichlet | `∏ xₖᵅᵏ⁻¹ / B(α)` |
| Poisson | `λˣ e^(−λ) / x!` |

### Information Theory

```
Entropy:         H(X) = −Σ p(x) log p(x)
Cross-entropy:   H(p,q) = −Σ p(x) log q(x)
KL divergence:   D_KL(P‖Q) = Σ p(x) log(p(x)/q(x))
Mutual info:     I(X;Y) = H(X) − H(X|Y)
```

> `D_KL ≥ 0` (Gibbs inequality). `D_KL(P‖Q) ≠ D_KL(Q‖P)` in general.

### MLE & MAP

```
MLE: θ* = argmax_θ  Σᵢ log P(xᵢ|θ)
MAP: θ* = argmax_θ  [log P(D|θ) + log P(θ)]
```

- Gaussian prior → **L2 regularization**  
- Laplace prior  → **L1 regularization**

---

## 4. ML Core — Loss Functions & Models

### Regression Losses

```
MSE:   L = (1/n) Σ(yᵢ − ŷᵢ)²
MAE:   L = (1/n) Σ|yᵢ − ŷᵢ|
Huber: L = ½(y−ŷ)²              if |e| ≤ δ
           δ|e| − ½δ²            otherwise
```

### Classification Losses

```
Log-loss: L = −(1/n) Σ[yᵢ log p̂ᵢ + (1−yᵢ) log(1−p̂ᵢ)]
Hinge:    L = max(0, 1 − yᵢ · ŷᵢ)          (SVM)
Focal:    L = −(1−p̂)ᵞ log(p̂)               (imbalanced, γ > 0)
```

### Regularization

```
L2 (Ridge):    L + λ‖θ‖₂²
L1 (Lasso):    L + λ‖θ‖₁
Elastic Net:   L + λ₁‖θ‖₁ + λ₂‖θ‖₂²
```

### Evaluation Metrics

```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 · P · R / (P + R)
R²        = 1 − SS_res / SS_tot
```

### Key Models

**Linear / Logistic Regression:**
```
Linear:   ŷ = wᵀx + b
Logistic: ŷ = σ(wᵀx + b),   σ(z) = 1/(1+e⁻ᶻ)
Softmax:  p(k|x) = exp(zₖ) / Σⱼ exp(zⱼ)
```

**SVM:**
```
Primal:  min ½‖w‖²  s.t. yᵢ(wᵀxᵢ + b) ≥ 1
Dual:    max Σαᵢ − ½ΣΣ αᵢαⱼ yᵢyⱼ xᵢᵀxⱼ
RBF:     K(xᵢ,xⱼ) = exp(−γ‖xᵢ−xⱼ‖²)
```

**PCA:**
```
1. Center:   x̃ = x − μ̄
2. Covariance: C = (1/n) X̃ᵀX̃
3. Eigen:    Cvₖ = λₖvₖ
4. Project:  z = Vₖᵀx̃   (Vₖ = top-k eigenvectors)

Explained variance ratio: λₖ / Σλ
```

**Bias–Variance Decomposition:**
```
E[(y − ŷ)²] = Bias² + Variance + σ²ₑ
Bias²       = (E[ŷ] − y)²
Variance    = E[(ŷ − E[ŷ])²]
```

---

## 5. Deep Learning

### Forward Pass

```
zˡ = Wˡ aˡ⁻¹ + bˡ
aˡ = f(zˡ)         (activation function)
ŷ  = aᴸ            (output layer)
```

### Backpropagation

```
Output delta:  δᴸ = ∇L ⊙ f'(zᴸ)
Hidden delta:  δˡ = (Wˡ⁺¹)ᵀ δˡ⁺¹ ⊙ f'(zˡ)

Gradients:
  ∂L/∂Wˡ = δˡ (aˡ⁻¹)ᵀ
  ∂L/∂bˡ = δˡ
```

`⊙` = Hadamard (element-wise product). O(params) per sample.

### Activation Functions

| Name | f(z) | f'(z) |
|------|------|-------|
| ReLU | `max(0, z)` | `𝟙[z > 0]` |
| Sigmoid | `1/(1+e⁻ᶻ)` | `σ(1−σ)` |
| Tanh | `(eᶻ−e⁻ᶻ)/(eᶻ+e⁻ᶻ)` | `1 − tanh²(z)` |
| GELU | `z · Φ(z)` | `Φ(z) + z·φ(z)` |
| Swish | `z · σ(z)` | `σ + z·σ(1−σ)` |
| ELU | `z if z≥0, α(eᶻ−1) if z<0` | `1 if z≥0, f(z)+α` |

### Normalization

**Batch Norm:**
```
μ_B = (1/m) Σ xᵢ
σ²_B = (1/m) Σ (xᵢ − μ_B)²
x̂ᵢ = (xᵢ − μ_B) / √(σ²_B + ε)
yᵢ = γ x̂ᵢ + β          (γ, β learned)
```

**Layer Norm:** normalize over features (not batch)  
**RMS Norm:** `x̂ᵢ = xᵢ / RMS(x)`,  `RMS = √(Σxᵢ²/d)`

### 2D Convolution

```
(I★K)[i,j] = Σₘ Σₙ I[i+m, j+n] · K[m,n]

Output size: H_out = ⌊(H + 2P − F) / S⌋ + 1
Parameters:  F² · Cᵢₙ · Cₒᵤₜ + Cₒᵤₜ  (with bias)
```

**Depthwise separable conv:**
```
FLOPs ≈ DW(CᵢK²H'W') + PW(CᵢCₒH'W')
Saving ≈ 1/Cₒ vs standard conv
```

---

## 6. Transformer Architecture

### Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V

Q = X·WQ,   K = X·WK,   V = X·WV
```

`√dₖ` scaling prevents softmax saturation. Complexity: `O(n²d)` per layer.

### Multi-Head Attention

```
headᵢ = Attention(Q·Wᵢᴼ, K·Wᵢᴷ, V·Wᵢᵛ)
MHA   = Concat(head₁, …, headₕ) · Wᴼ
```

`h` heads, `dₖ = d_model / h`

### Positional Encodings

**Sinusoidal (original):**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

**RoPE (Rotary Position Embedding):**
```
Rotate query/key by angle mθ per dimension pair.
Used in LLaMA, Mistral, GPT-4 family.
```

### Feed-Forward & Residual Block

```
FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂
d_ffn  = 4 · d_model  (typical)

Residual (pre-LN): x ← x + SubLayer(LN(x))
```

### Efficient Attention Variants

| Method | Key Idea |
|---|---|
| FlashAttention | `O(n²)` time, `O(n)` HBM I/O via tiling |
| Linear Attention | `Attn ≈ φ(Q)(φ(K)ᵀV)` — `O(n)` |
| MQA | 1 shared KV head |
| GQA | G shared KV groups (`h_KV < h_Q`) |

### Training Objectives

```
CLM (GPT):    L = −Σ log P(xₜ | x₁,…,xₜ₋₁)
MLM (BERT):   L = −Σ log P(x_mask | x_visible)
RLHF:         r(x,y) − β · KL(π ‖ π_ref)
```

---

## 7. Generative Models

### VAE (Variational Autoencoder)

```
ELBO = E_q[log p(x|z)] − D_KL(q(z|x) ‖ p(z))

Reparameterization: z = μ + σ ⊙ ε,   ε ~ 𝒩(0,I)

KL (Gaussians): −½ Σ(1 + log σ² − μ² − σ²)
```

### GAN (Generative Adversarial Network)

```
Original: min_G max_D  E[log D(x)] + E[log(1 − D(G(z)))]

Wasserstein: min_G max_{‖f‖_L≤1}  E[f(x)] − E[f(G(z))]
```

### Diffusion Models (DDPM)

```
Forward (noising): q(xₜ|xₜ₋₁) = 𝒩(√(1−βₜ)xₜ₋₁,  βₜI)
Shortcut:          q(xₜ|x₀)   = 𝒩(√ᾱₜ x₀,  (1−ᾱₜ)I)
                   ᾱₜ = ∏ₛ(1−βₛ)

Loss (noise prediction): L = E‖ε − εθ(xₜ, t)‖²
```

DDIM enables deterministic (fewer-step) sampling.

### Flow Matching / Rectified Flow

```
ODE:  dx/dt = v_θ(x, t)
Loss: E‖v_θ(tX₁ + (1−t)X₀, t) − (X₁ − X₀)‖²
```

Straight trajectories; used in Stable Diffusion 3, Flux.

---

## 8. NLP

### Language Modeling

**N-gram:**
```
P(w₁,…,wₙ) = ∏ P(wᵢ | wᵢ₋ₙ₊₁,…,wᵢ₋₁)

Perplexity: PP = P(w₁…wₙ)^(−1/N)   (lower = better)
```

**Word2Vec (skip-gram):**
```
Objective: max Σ Σ log P(wₒ | wc)
P(wₒ|wc)  = exp(uₒᵀvc) / Σⱼ exp(uⱼᵀvc)
```

**GloVe:**
```
J = Σᵢⱼ f(Xᵢⱼ)(wᵢᵀw̃ⱼ + bᵢ + b̃ⱼ − log Xᵢⱼ)²
```

### Tokenization

```
BPE:          merge most frequent byte pair iteratively
SentencePiece: unigram LM — prune segments with small loss drop
```

### Sequence-to-Sequence + Attention

```
Encoder: h = f_enc(x₁,…,xₙ)
Decoder: P(y|x) = ∏ P(yₜ | y<ₜ, h)

Bahdanau attention:
  eₜᵢ = vᵀ tanh(Wsₜ₋₁ + Uhᵢ)
  αₜᵢ = softmax(eₜᵢ)
  cₜ  = Σᵢ αₜᵢ hᵢ
```

### Fine-tuning & Adaptation

**LoRA (Low-Rank Adaptation):**
```
W' = W + ΔW = W + B·A
B ∈ ℝᵈˣʳ,  A ∈ ℝʳˣᵏ,  r ≪ min(d, k)
```
Typical rank r = 4–64; saves 10–100× parameters vs full fine-tuning.

### Decoding Strategies

```
Greedy:       ŷₜ = argmax P(yₜ | y<ₜ)
Beam search:  keep top-B partial sequences
Temperature:  p'(w) = p(w)^(1/T) / Z     (T→0: greedy, T→∞: uniform)
Top-p:        sample from smallest set where ΣP ≥ p
```

### RLHF & Alignment

**PPO:**
```
Reward: r(x,y) = r_ψ(x,y) − β log[π_θ(y|x) / π_ref(y|x)]
L_PPO  = E[min(rₜ(θ)·Aₜ,  clip(rₜ, 1±ε)·Aₜ)]
```

**DPO (closed form, no RL):**
```
L = −E[log σ(β log πθ(y_w|x)/π_ref(y_w|x)
         − β log πθ(y_l|x)/π_ref(y_l|x))]
```

### Text Similarity & Retrieval

```
BM25: Σ IDF(qᵢ) · tf·(k₁+1) / (tf + k₁(1−b + b·dl/avgdl))
BLEU: BP · exp(Σ wₙ log pₙ)
ROUGE-N: |S∩R| / |R|    (n-gram recall)

Contrastive (InfoNCE):
  L = −log exp(q·k⁺/τ) / Σᵢ exp(q·kᵢ/τ)
```

---

## 9. Computer Vision

### Image Representation

```
Tensor: I ∈ ℝᴴˣᵂˣᶜ

Normalization: x̂ = (x − μ) / σ
ImageNet:  μ = [0.485, 0.456, 0.406]
           σ = [0.229, 0.224, 0.225]

Batch (PyTorch): N×C×H×W
Batch (TF/Keras): N×H×W×C
```

### Convolution Geometry

```
H_out = ⌊(H + 2P − K) / S⌋ + 1

Same padding (S=1): P = (K−1)/2
Transposed conv:    H_out = (H−1)·S − 2P + K

FLOPs per layer ≈ 2 · Cᵢ · Cₒ · K² · H' · W'
```

### Spatial Transforms (Affine)

```
[x']   [a  b  c] [x]
[y'] = [d  e  f] [y]
[1 ]   [0  0  1] [1]

Rotation matrix: [[cosθ, −sinθ], [sinθ, cosθ]]
```

### Object Detection

**Bounding box encoding (anchor-based):**
```
tₓ = (x − xₐ)/wₐ,   tᵧ = (y − yₐ)/hₐ
t_w = log(w/wₐ),    t_h = log(h/hₐ)
```

**Overlap metrics:**
```
IoU  = |A∩B| / |A∪B|
GIoU = IoU − |C\(A∪B)| / |C|      (C = enclosing box)
```

**Detection loss:**
```
L = L_cls + λ · L_reg

Smooth L1: ½x²           if |x| < 1
           |x| − 0.5     otherwise

mAP: mean AP over classes (and IoU thresholds)
AP  = ∫₀¹ p(r) dr
```

**Non-Maximum Suppression (NMS):**
```
1. Sort detections by score ↓
2. Keep highest-score box
3. Remove boxes with IoU > threshold
4. Repeat

Soft-NMS: sᵢ ← sᵢ · exp(−IoU²/σ)   (decay, not remove)
```

**Anchor-free (FCOS):**
```
Predict (l, t, r, b) — distances to box edges

centerness = √( min(l,r)/max(l,r) · min(t,b)/max(t,b) )
```

### Segmentation

```
Pixel Accuracy = Σ nᵢᵢ / Σ tᵢ
Mean IoU       = (1/k) Σ nᵢᵢ / (tᵢ + Σⱼnⱼᵢ − nᵢᵢ)
Dice           = 2|A∩B| / (|A| + |B|)
Dice Loss      = 1 − 2Σpᵢgᵢ / (Σpᵢ + Σgᵢ + ε)
```

### Vision Transformer (ViT)

```
Patches:  N = HW/P²   (P×P patch size)
Embed:    z₀ = [x_cls; x_p¹E; …; x_pᴺE] + E_pos
Encoder:  standard transformer blocks
Output:   y = MLP(z_class)

Attention complexity: O((HW/P²)²)   vs O(HW) for CNNs
```

### CLIP — Contrastive Vision-Language

```
Similarity: sᵢⱼ = enc_img(xᵢ) · enc_txt(yⱼ) / (‖…‖ · ‖…‖)

Loss (N×N matrix, diagonal = positives):
L = −(1/2N)[Σᵢ log exp(sᵢᵢ/τ)/Σⱼ exp(sᵢⱼ/τ)
           + Σⱼ log exp(sⱼⱼ/τ)/Σᵢ exp(sᵢⱼ/τ)]
```

### Optical Flow & Depth

```
Brightness constancy: I(x,y,t) = I(x+u, y+v, t+1)

Stereo depth: Z = f · B / d
  f = focal length,  B = baseline,  d = disparity

Log-depth loss: L = E|log d̂ − log d*|
```

---

*All formulas use standard notation. Subscripts denote layer (superscript l), time step (subscript t), or index (subscript i/j/k).*
