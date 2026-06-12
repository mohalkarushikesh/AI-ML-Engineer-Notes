# PyTorch Cheatsheet

---

## Installation

```bash
# CUDA (GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision torchaudio

# Check version & GPU
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

---

## Tensors

### Creating Tensors
```python
import torch

torch.tensor([1, 2, 3])                  # From Python list
torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # 2D tensor

torch.zeros(3, 4)                        # All zeros
torch.ones(3, 4)                         # All ones
torch.full((3, 4), 7.0)                  # Fill with value
torch.eye(4)                             # Identity matrix
torch.empty(3, 4)                        # Uninitialized

torch.rand(3, 4)                         # Uniform [0, 1)
torch.randn(3, 4)                        # Standard normal
torch.randint(0, 10, (3, 4))            # Random integers

torch.arange(0, 10, 2)                  # [0, 2, 4, 6, 8]
torch.linspace(0, 1, 5)                 # [0, .25, .5, .75, 1]

torch.zeros_like(x)                     # Same shape/dtype as x
torch.ones_like(x)
torch.rand_like(x)
```

### Tensor Properties
```python
x = torch.randn(3, 4)

x.shape          # torch.Size([3, 4])
x.size()         # Same as shape
x.ndim           # 2
x.dtype          # torch.float32
x.device         # device('cpu')
x.requires_grad  # False
x.numel()        # 12 (total elements)
x.is_cuda        # False
```

### Data Types
| dtype | Description |
|-------|-------------|
| `torch.float32` | Default float (fp32) |
| `torch.float16` | Half precision (fp16) |
| `torch.bfloat16` | Brain float (bf16, better range) |
| `torch.float64` | Double precision |
| `torch.int32` | 32-bit integer |
| `torch.int64` | 64-bit integer (default int) |
| `torch.bool` | Boolean |

```python
x = x.float()       # Cast to float32
x = x.half()        # Cast to float16
x = x.to(torch.bfloat16)
x = x.long()        # Cast to int64
```

---

## Tensor Operations

### Shape Manipulation
```python
x = torch.randn(2, 3, 4)

x.reshape(6, 4)          # Reshape (may copy)
x.view(6, 4)             # Reshape (contiguous only)
x.flatten()              # → 1D tensor
x.flatten(1)             # Flatten from dim 1 onwards
x.squeeze()              # Remove dims of size 1
x.squeeze(0)             # Remove specific dim if size 1
x.unsqueeze(0)           # Add dim at position 0 → (1, 2, 3, 4)

x.permute(2, 0, 1)       # Reorder dimensions
x.transpose(0, 1)        # Swap two dimensions
x.contiguous()           # Make memory contiguous
```

### Indexing & Slicing
```python
x = torch.randn(4, 5)

x[0]             # First row
x[:, 1]          # Second column
x[1:3, 2:4]      # Slice rows 1-2, cols 2-3
x[x > 0]         # Boolean mask → 1D
x[[0, 2], :]     # Fancy indexing

# Advanced
torch.where(x > 0, x, torch.zeros_like(x))   # Conditional select
torch.gather(x, dim=1, index=idx)             # Gather along dim
torch.scatter_(x, dim=1, index=idx, src=val) # Scatter update
```

### Math Operations
```python
# Element-wise
x + y,  x - y,  x * y,  x / y    # Arithmetic
x ** 2,  torch.sqrt(x)
torch.abs(x),  torch.exp(x),  torch.log(x)
torch.clamp(x, min=0, max=1)      # Clip values

# Matrix ops
x @ y                # Matrix multiply (preferred)
torch.mm(x, y)       # 2D matrix multiply
torch.bmm(x, y)      # Batched matrix multiply (3D)
torch.matmul(x, y)   # General (handles broadcasting)

# Reductions
x.sum()              # Sum all
x.sum(dim=0)         # Sum along dim, keeps shape if keepdim=True
x.mean(), x.std(), x.var()
x.max(), x.min()
x.argmax(), x.argmin()
x.max(dim=1)         # Returns (values, indices)
torch.topk(x, k=3)  # Top-k values & indices

# Comparison
x == y,  x > y,  x >= y
torch.equal(x, y)    # True if all elements equal
```

### Concatenation & Stacking
```python
torch.cat([a, b, c], dim=0)    # Concatenate along existing dim
torch.stack([a, b, c], dim=0)  # Stack along NEW dim
torch.chunk(x, chunks=3, dim=0) # Split into N chunks
torch.split(x, split_size=2, dim=0) # Split by size
```

---

## Device Management

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move tensors
x = x.to(device)
x = x.cuda()       # To GPU
x = x.cpu()        # To CPU

# GPU info
torch.cuda.device_count()
torch.cuda.current_device()
torch.cuda.get_device_name(0)
torch.cuda.memory_allocated()    # Bytes used
torch.cuda.empty_cache()         # Free cached memory

# MPS (Apple Silicon)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

---

## Autograd & Gradients

```python
x = torch.randn(3, requires_grad=True)
y = (x ** 2).sum()
y.backward()         # Compute gradients
x.grad               # ∂y/∂x

# Stop gradient tracking
with torch.no_grad():
    y = model(x)     # No gradients computed (inference)

x.detach()           # New tensor, no gradient
x.detach_()          # In-place detach

# Retain graph for multiple backward passes
y.backward(retain_graph=True)

# Gradient manipulation
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)

# Zero gradients before each backward
optimizer.zero_grad()   # Or set_to_none=True for speed
```

---

## Building Models

### nn.Module
```python
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.3)
        self.bn = nn.BatchNorm1d(256)

    def forward(self, x):
        x = F.relu(self.bn(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

model = MyModel().to(device)
```

### nn.Sequential
```python
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 10)
)
```

### Common Layers
```python
# Linear
nn.Linear(in, out, bias=True)

# Convolutional
nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)  # Upsample

# Pooling
nn.MaxPool2d(kernel_size=2, stride=2)
nn.AvgPool2d(kernel_size=2)
nn.AdaptiveAvgPool2d((1, 1))    # Global average pool

# Normalization
nn.BatchNorm1d(num_features)
nn.BatchNorm2d(num_features)
nn.LayerNorm(normalized_shape)
nn.GroupNorm(num_groups, num_channels)

# Recurrent
nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

# Attention
nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1)

# Embedding
nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)

# Dropout
nn.Dropout(p=0.5)
nn.Dropout2d(p=0.5)   # Drops entire channels
```

### Model Inspection
```python
# Parameter count
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

# Print architecture
print(model)
from torchinfo import summary
summary(model, input_size=(32, 1, 28, 28))

# Named parameters
for name, param in model.named_parameters():
    print(name, param.shape)

# Freeze / unfreeze
for param in model.parameters():
    param.requires_grad = False   # Freeze all
model.fc3.requires_grad_(True)    # Unfreeze head
```

---

## Loss Functions

```python
nn.MSELoss()                           # Regression
nn.L1Loss()                            # MAE
nn.SmoothL1Loss()                      # Huber loss

nn.BCELoss()                           # Binary CE (after sigmoid)
nn.BCEWithLogitsLoss()                 # Binary CE (includes sigmoid)
nn.CrossEntropyLoss()                  # Multi-class (includes softmax)
nn.NLLLoss()                           # Negative log-likelihood

nn.CrossEntropyLoss(weight=class_wts)  # Weighted (imbalanced classes)
nn.CrossEntropyLoss(label_smoothing=0.1)

nn.KLDivLoss(reduction='batchmean')    # KL divergence
nn.CosineEmbeddingLoss()               # Similarity learning
nn.TripletMarginLoss()                 # Metric learning
```

---

## Optimizers

```python
import torch.optim as optim

optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
optim.RMSprop(model.parameters(), lr=1e-3)

# Per-layer learning rates
optimizer = optim.Adam([
    {'params': model.backbone.parameters(), 'lr': 1e-4},
    {'params': model.head.parameters(),     'lr': 1e-3},
])
```

### LR Schedulers
```python
from torch.optim import lr_scheduler

lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)
lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(loader), epochs=10)

# Step the scheduler
scheduler.step()                  # Every epoch
scheduler.step(val_loss)          # ReduceLROnPlateau only
```

---

## Data Loading

```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.transform:
            x = self.transform(x)
        return x, self.y[idx]

# DataLoader
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,       # Parallel loading
    pin_memory=True,     # Faster GPU transfer
    drop_last=False,     # Drop incomplete last batch
    prefetch_factor=2,   # Batches prefetched per worker
)

# Iterate
for X, y in loader:
    X, y = X.to(device), y.to(device)
```

### Transforms (torchvision)
```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),                            # PIL → [0,1] tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])  # ImageNet stats
])
```

---

## Training Loop

```python
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        correct += (logits.argmax(1) == y).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        total_loss += loss.item() * X.size(0)
        correct += (logits.argmax(1) == y).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

# Main loop
for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    scheduler.step()
    print(f"Epoch {epoch+1:03d} | Train {train_loss:.4f}/{train_acc:.3f} | Val {val_loss:.4f}/{val_acc:.3f}")
```

---

## Saving & Loading

```python
# Save full model (not recommended for sharing)
torch.save(model, 'model.pt')
model = torch.load('model.pt')

# Save state dict (recommended)
torch.save(model.state_dict(), 'weights.pth')
model.load_state_dict(torch.load('weights.pth', map_location=device))

# Save full checkpoint
checkpoint = {
    'epoch': epoch,
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
    'val_loss': val_loss,
}
torch.save(checkpoint, 'checkpoint.pth')

# Load checkpoint
ckpt = torch.load('checkpoint.pth', map_location=device)
model.load_state_dict(ckpt['model'])
optimizer.load_state_dict(ckpt['optimizer'])
start_epoch = ckpt['epoch'] + 1
```

---

## Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for X, y in loader:
    X, y = X.to(device), y.to(device)
    optimizer.zero_grad(set_to_none=True)

    with autocast():                   # fp16 forward pass
        logits = model(X)
        loss = criterion(logits, y)

    scaler.scale(loss).backward()      # Scale gradients
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
```

---

## Transfer Learning

```python
import torchvision.models as models

# Load pretrained model
model = models.resnet50(weights='IMAGENET1K_V2')

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace head for new task
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Fine-tune with different LRs
optimizer = optim.AdamW([
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(),     'lr': 1e-3},
], weight_decay=0.01)
```

### Available torchvision Models
```python
models.resnet50(), models.resnet101()
models.efficientnet_b0(), models.efficientnet_v2_s()
models.vit_b_16(), models.vit_l_16()
models.mobilenet_v3_large()
models.convnext_tiny(), models.convnext_base()
models.swin_t(), models.swin_b()
```

---

## Multi-GPU Training (DDP)

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Launch: torchrun --nproc_per_node=4 train.py

dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
device = torch.device(f'cuda:{local_rank}')

model = MyModel().to(device)
model = DDP(model, device_ids=[local_rank])

# Use DistributedSampler
sampler = torch.utils.data.distributed.DistributedSampler(dataset)
loader = DataLoader(dataset, batch_size=32, sampler=sampler)

# Cleanup
dist.destroy_process_group()
```

---

## Inference & Deployment

```python
# Standard inference
model.eval()
with torch.no_grad():
    output = model(input_tensor)
    probs = torch.softmax(output, dim=-1)
    pred = probs.argmax(dim=-1)

# torch.compile (PyTorch 2+, faster inference)
model = torch.compile(model)

# Export to ONNX
torch.onnx.export(
    model, dummy_input, "model.onnx",
    input_names=['input'], output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}}
)

# TorchScript (serializable, no Python needed)
scripted = torch.jit.script(model)
scripted.save("model_scripted.pt")
loaded = torch.jit.load("model_scripted.pt")

# Quantization (CPU inference speedup)
model_int8 = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)
```

---

## Useful Utilities

```python
# Reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False   # True for fixed input sizes = faster

# Tensor ↔ NumPy
arr = tensor.cpu().numpy()         # Tensor → NumPy (shares memory)
t = torch.from_numpy(arr)          # NumPy → Tensor (shares memory)
t = torch.tensor(arr)              # NumPy → Tensor (copy)

# Profiling
with torch.profiler.profile(activities=[
    torch.profiler.ProfilerActivity.CPU,
    torch.profiler.ProfilerActivity.CUDA,
]) as prof:
    model(x)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Functional API
F.relu(x), F.gelu(x), F.sigmoid(x)
F.softmax(x, dim=-1), F.log_softmax(x, dim=-1)
F.cross_entropy(logits, targets)
F.binary_cross_entropy_with_logits(logits, targets)
F.normalize(x, p=2, dim=-1)        # L2 normalize
F.cosine_similarity(x1, x2)
F.pad(x, pad=(1, 1, 1, 1))         # Pad (left, right, top, bottom)
```

---

## Debugging Tips

```python
# Detect NaN/Inf
torch.isnan(x).any()
torch.isinf(x).any()

# Anomaly detection (find where NaN first appears)
torch.autograd.set_detect_anomaly(True)

# Shape debugging
print(f"x: {x.shape}, y: {y.shape}")  # Add everywhere

# Check gradient flow
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm():.4f}")

# Memory debugging
print(torch.cuda.memory_summary())
```

---

## Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `RuntimeError: Expected all tensors on same device` | Mixed CPU/GPU | `.to(device)` all tensors |
| `RuntimeError: size mismatch` | Wrong tensor shapes | Check with `.shape` |
| `loss is nan` | Exploding gradients, bad lr | Clip gradients, lower lr |
| `CUDA out of memory` | Batch too large | Reduce batch size, use gradient accumulation |
| `inplace operation error` | Autograd conflict | Avoid `x += y`, use `x = x + y` |
| `Expected contiguous tensor` | Non-contiguous view | Call `.contiguous()` |
| `RuntimeError: cudnn error` | cuDNN version mismatch | Reinstall matching PyTorch/CUDA |

---

*Last updated: 2025 · Tested on PyTorch 2.x · Always pin package versions in production.*
