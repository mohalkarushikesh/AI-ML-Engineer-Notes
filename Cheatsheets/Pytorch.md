check the official [PyTorch Cheat Sheet](https://docs-preview.pytorch.org/pytorch/tutorials/2482/beginner/ptcheat.html) or the detailed version from [Sling Academy](https://www.slingacademy.com/article/pytorch-complete-cheat-sheet/).

---

## 🔑 Imports
```python
import torch                          # core package
from torch.utils.data import Dataset, DataLoader
import torch.autograd as autograd     # automatic differentiation
from torch import Tensor
import torch.nn as nn                 # neural networks
import torch.nn.functional as F       # layers, activations
import torch.optim as optim           # optimizers
```

---

## 📦 Tensors
- **Create tensors**
```python
x = torch.tensor([1, 2, 3])
x = torch.zeros(2, 3)     # 2x3 matrix of zeros
x = torch.ones(2, 3)      # 2x3 matrix of ones
x = torch.rand(2, 3)      # random values
```

- **Operations**
```python
y = x.view(-1)            # reshape
z = x + y                 # addition
z = x @ y.T               # matrix multiplication
```

---

## ⚡ Autograd
```python
x = torch.ones(2, 2, requires_grad=True)
y = x + 2
z = y * y * 3
out = z.mean()
out.backward()            # compute gradients
print(x.grad)             # gradients w.r.t x
```

---

## 🧠 Neural Networks
```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 5)

    def forward(self, x):
        return F.relu(self.fc1(x))

net = Net()
```

---

## 🎯 Loss & Optimization
```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

# Training step
optimizer.zero_grad()
output = net(torch.randn(1, 10))
loss = criterion(output, torch.tensor([1]))
loss.backward()
optimizer.step()
```

---

## 📊 Data Loading
```python
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = net(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

---

## 🚀 Extras
- **TorchScript/JIT**: `torch.jit.trace()` and `@torch.jit.script` for model optimization.  
- **ONNX Export**: `torch.onnx.export(model, inputs, "model.onnx")`.  

---
