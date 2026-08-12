# ML / DL Engineer — Hands-On Coding Workbook

Practical exercises to build syntax fluency. Attempt each **before** reading the solution.
Stack: **NumPy · scikit-learn · PyTorch** (industry-standard for DL/ML engineering).

Tiers:
🟢 **Beginner** — get the syntax working
🟡 **Medium** — real workflow, small design decisions
🔴 **Advanced (Legend / Aura++)** — implement internals from scratch / production-grade

Setup:
```bash
pip install numpy scikit-learn torch torchvision matplotlib
```

---

## TOPIC 1 — NumPy / Tensor Fundamentals

### 🟢 Beginner
**Problem:** Create a 4×4 matrix of numbers 1–16, extract the second row, and compute the mean of each column.
```python
import numpy as np
a = np.arange(1, 17).reshape(4, 4)
print(a[1])            # second row
print(a.mean(axis=0))  # mean of each column
```

### 🟡 Medium
**Problem:** Given a batch of 100 images shaped `(100, 28, 28)`, normalize each image to mean 0, std 1 (per image), and flatten to `(100, 784)`.
```python
X = np.random.rand(100, 28, 28)
mean = X.mean(axis=(1, 2), keepdims=True)
std  = X.std(axis=(1, 2), keepdims=True) + 1e-8
X_norm = (X - mean) / std
X_flat = X_norm.reshape(100, -1)   # (100, 784)
```

### 🔴 Advanced (Legend)
**Problem:** Implement softmax over a 2D array (rows = samples) **numerically stably** without loops.
```python
def softmax(z):
    z = z - z.max(axis=1, keepdims=True)   # stability: subtract row max
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
# Why subtract max? exp of large numbers overflows; shifting doesn't change the result.
```

---

## TOPIC 2 — Classic ML with scikit-learn

### 🟢 Beginner
**Problem:** Train a logistic regression on the Iris dataset and print test accuracy.
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X, y = load_iris(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
clf = LogisticRegression(max_iter=200).fit(Xtr, ytr)
print("accuracy:", clf.score(Xte, yte))
```

### 🟡 Medium
**Problem:** Build a **Pipeline** that scales features then fits a Random Forest, and evaluate with 5-fold cross-validation.
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(n_estimators=200, random_state=42)),
])
scores = cross_val_score(pipe, X, y, cv=5)
print("cv mean:", scores.mean(), "±", scores.std())
```

### 🔴 Advanced (Aura++)
**Problem:** Tune hyperparameters with `GridSearchCV` on an imbalanced dataset, optimizing for **F1** (not accuracy), and report the best params + a full classification report.
```python
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

param_grid = {"rf__n_estimators": [100, 300], "rf__max_depth": [None, 10, 20]}
grid = GridSearchCV(pipe, param_grid, scoring="f1_macro", cv=5, n_jobs=-1)
grid.fit(Xtr, ytr)
print("best params:", grid.best_params_)
print(classification_report(yte, grid.predict(Xte)))
```

---

## TOPIC 3 — Linear/Logistic Regression FROM SCRATCH

### 🟢 Beginner
**Problem:** Implement the sigmoid and the binary cross-entropy loss.
```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def bce(y, p, eps=1e-9):
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
```

### 🟡 Medium
**Problem:** Implement one step of gradient descent for logistic regression (weights `w`, bias `b`).
```python
def grad_step(X, y, w, b, lr=0.1):
    m = X.shape[0]
    p = sigmoid(X @ w + b)
    dw = (X.T @ (p - y)) / m
    db = np.mean(p - y)
    return w - lr * dw, b - lr * db
```

### 🔴 Advanced (Legend)
**Problem:** Train full logistic regression from scratch on a toy dataset and confirm loss decreases.
```python
np.random.seed(0)
X = np.random.randn(200, 3)
true_w = np.array([2.0, -1.0, 0.5])
y = (sigmoid(X @ true_w) > 0.5).astype(float)

w, b = np.zeros(3), 0.0
for epoch in range(1000):
    w, b = grad_step(X, y, w, b, lr=0.5)
    if epoch % 200 == 0:
        print(epoch, bce(y, sigmoid(X @ w + b)))
print("learned w:", w)   # should approach direction of true_w
```

---

## TOPIC 4 — Neural Networks in PyTorch (CORE)

### 🟢 Beginner
**Problem:** Define a 2-layer MLP for 10-class classification on 784-dim input.
```python
import torch, torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)
x = torch.randn(32, 784)   # batch of 32
print(model(x).shape)      # torch.Size([32, 10])
```

### 🟡 Medium
**Problem:** Write a complete training loop (loss, backward, optimizer step) for one epoch.
```python
import torch.nn.functional as F

loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

def train_one_epoch(loader):
    model.train()
    for xb, yb in loader:
        opt.zero_grad()          # clear old gradients
        out = model(xb)
        loss = loss_fn(out, yb)
        loss.backward()          # compute gradients
        opt.step()               # update weights
    return loss.item()
```
**Common interview gotcha:** forgetting `opt.zero_grad()` → gradients accumulate across steps.

### 🔴 Advanced (Aura++max)
**Problem:** Build a custom `nn.Module` with dropout + batch norm, plus a train/val loop that tracks best val accuracy and does early stopping.
```python
class Net(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, p=0.3):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.bn  = nn.BatchNorm1d(hidden)
        self.drop = nn.Dropout(p)
        self.fc2 = nn.Linear(hidden, out_dim)
    def forward(self, x):
        x = F.relu(self.bn(self.fc1(x)))
        x = self.drop(x)
        return self.fc2(x)

def fit(model, tr_loader, va_loader, epochs=50, patience=5):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_acc, wait = 0.0, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            opt.zero_grad(); loss = loss_fn(model(xb), yb)
            loss.backward(); opt.step()
        # validation
        model.eval(); correct = total = 0
        with torch.no_grad():
            for xb, yb in va_loader:
                pred = model(xb).argmax(1)
                correct += (pred == yb).sum().item(); total += yb.size(0)
        acc = correct / total
        if acc > best_acc:
            best_acc, wait = acc, 0
            torch.save(model.state_dict(), "best.pt")   # checkpoint
        else:
            wait += 1
            if wait >= patience:
                print(f"early stop @ epoch {ep}"); break
    return best_acc
```
Note `model.train()` vs `model.eval()` — toggles dropout/batchnorm behavior. Forgetting this is a classic bug.

---

## TOPIC 5 — CNNs in PyTorch

### 🟢 Beginner
**Problem:** Define a single conv layer and check output shape for a 3-channel 32×32 image.
```python
conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
x = torch.randn(1, 3, 32, 32)
print(conv(x).shape)   # torch.Size([1, 16, 32, 32])  (padding keeps HxW)
```

### 🟡 Medium
**Problem:** Build a small CNN (conv → relu → pool → conv → relu → pool → fc) for 10-class 32×32 images.
```python
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool  = nn.MaxPool2d(2)
        self.fc    = nn.Linear(32 * 8 * 8, 10)   # 32->16->8 after two pools
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(1)
        return self.fc(x)
```
**Interview math:** know how to compute output size: `out = (in + 2*pad - kernel)/stride + 1`.

### 🔴 Advanced (Legend)
**Problem:** Implement a **residual block** (the ResNet building block) from scratch.
```python
class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(ch)
    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity          # <-- the skip connection
        return F.relu(out)
```

---

## TOPIC 6 — RNN / LSTM

### 🟢 Beginner
**Problem:** Run an LSTM over a sequence and read its output shapes.
```python
lstm = nn.LSTM(input_size=10, hidden_size=20, batch_first=True)
x = torch.randn(4, 7, 10)          # (batch, seq_len, features)
out, (h, c) = lstm(x)
print(out.shape, h.shape)          # (4,7,20)  (1,4,20)
```

### 🟡 Medium
**Problem:** Build an LSTM classifier that uses the **last timestep** for prediction.
```python
class LSTMClassifier(nn.Module):
    def __init__(self, in_dim, hid, n_classes):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hid, batch_first=True)
        self.fc = nn.Linear(hid, n_classes)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])   # last timestep
```

### 🔴 Advanced (Aura++)
**Problem:** Handle variable-length sequences with padding + `pack_padded_sequence`.
```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def forward_packed(lstm, x, lengths):
    packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                  enforce_sorted=False)
    out_packed, (h, c) = lstm(packed)
    out, _ = pad_packed_sequence(out_packed, batch_first=True)
    return out, h
# Packing skips computation on pad tokens -> faster & correct hidden states.
```

---

## TOPIC 7 — Regularization in Code

### 🟢 Beginner
**Problem:** Add dropout and L2 (weight decay) to a model + optimizer.
```python
model = nn.Sequential(nn.Linear(100, 50), nn.ReLU(), nn.Dropout(0.5), nn.Linear(50, 2))
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)  # L2
```

### 🟡 Medium
**Problem:** Add an explicit **L1 penalty** to the loss manually.
```python
def l1_penalty(model, lam=1e-4):
    return lam * sum(p.abs().sum() for p in model.parameters())

loss = loss_fn(model(xb), yb) + l1_penalty(model)
loss.backward()
```

### 🔴 Advanced (Legend)
**Problem:** Implement **label smoothing** cross-entropy from scratch (regularizes overconfident predictions).
```python
def smooth_ce(logits, target, n_classes, eps=0.1):
    logp = F.log_softmax(logits, dim=1)
    with torch.no_grad():
        true_dist = torch.full_like(logp, eps / (n_classes - 1))
        true_dist.scatter_(1, target.unsqueeze(1), 1 - eps)
    return torch.mean(torch.sum(-true_dist * logp, dim=1))
```

---

## TOPIC 8 — Transformers / Self-Attention (HIGH VALUE)

### 🟢 Beginner
**Problem:** Use PyTorch's built-in multi-head attention.
```python
mha = nn.MultiheadAttention(embed_dim=64, num_heads=8, batch_first=True)
x = torch.randn(2, 10, 64)              # (batch, seq, embed)
out, weights = mha(x, x, x)             # self-attention: Q=K=V=x
print(out.shape)                        # (2, 10, 64)
```

### 🟡 Medium
**Problem:** Implement **scaled dot-product attention** from scratch.
```python
import math
def attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)   # the √dₖ scaling
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return weights @ V, weights
```

### 🔴 Advanced (Aura++maxmax)
**Problem:** Build a full **multi-head attention module** from scratch (no `nn.MultiheadAttention`).
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.d_k = n_heads, d_model // n_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
    def forward(self, x):
        B, T, _ = x.shape
        def split(t): return t.view(B, T, self.h, self.d_k).transpose(1, 2)
        Q, K, V = split(self.wq(x)), split(self.wk(x)), split(self.wv(x))
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)
```

---

## TOPIC 9 — Deployment / MLOps (THE ML-ENGINEER DIFFERENTIATOR)

### 🟢 Beginner
**Problem:** Save and load a model correctly for inference.
```python
torch.save(model.state_dict(), "model.pt")      # save weights only (preferred)
model.load_state_dict(torch.load("model.pt"))
model.eval()                                     # ALWAYS before inference
```

### 🟡 Medium
**Problem:** Wrap a model in a minimal FastAPI inference endpoint.
```python
from fastapi import FastAPI
import torch

app = FastAPI()
model.eval()

@app.post("/predict")
def predict(features: list[float]):
    x = torch.tensor([features], dtype=torch.float32)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)
    return {"class": int(probs.argmax()), "confidence": float(probs.max())}
```

### 🔴 Advanced (Legend / Aura++)
**Problem:** Export to ONNX and apply dynamic quantization to shrink the model for serving.
```python
# Export to ONNX (framework-agnostic serving format)
dummy = torch.randn(1, 784)
torch.onnx.export(model, dummy, "model.onnx",
                  input_names=["input"], output_names=["output"],
                  dynamic_axes={"input": {0: "batch"}})

# Dynamic quantization: FP32 -> INT8 on Linear layers (smaller, faster CPU inference)
quantized = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8)
torch.save(quantized.state_dict(), "model_int8.pt")
```
System-design follow-ups to rehearse: batching requests, GPU vs CPU serving, latency vs throughput tradeoffs, model versioning, monitoring for data drift.

---

## HOW TO USE THIS

1. Work top-to-bottom per topic — beginner cements syntax, medium builds the workflow, advanced proves you understand internals.
2. **Type it, don't copy-paste.** Muscle memory for syntax is what interviews test under pressure.
3. For each advanced exercise, be ready to *explain why* — e.g. why √dₖ, why `zero_grad()`, why `eval()` matters.
4. Priority order for ML/DL engineer: **Topic 4 (PyTorch core) → 5 (CNN) → 8 (attention) → 9 (deployment)**, then the rest.

*Want a live coding drill? Ask me and I'll give you a blank problem, you paste your attempt, and I'll review it like an interviewer.*
