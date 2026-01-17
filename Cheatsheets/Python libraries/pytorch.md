## 📌 PyTorch Cheat Sheet

### 🔹 Imports
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
```

---

### 🔹 Tensors
- **Create tensors**
```python
x = torch.tensor([1, 2, 3])
x = torch.zeros(2, 3)       # 2x3 matrix of zeros
x = torch.ones(2, 3)        # 2x3 matrix of ones
x = torch.rand(2, 3)        # random values
```

- **Operations**
```python
y = x.view(-1)              # reshape
z = x + y                   # addition
z = x @ y                   # matrix multiplication
```

- **Device**
```python
x = torch.rand(3, 3).to("cuda")   # move to GPU
```

---

### 🔹 Autograd (Automatic Differentiation)
```python
x = torch.ones(2, 2, requires_grad=True)
y = x + 2
z = y * y * 3
out = z.mean()
out.backward()              # compute gradients
print(x.grad)               # gradients w.r.t x
```

---

### 🔹 Neural Networks
```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 5)   # fully connected layer

    def forward(self, x):
        return F.relu(self.fc1(x))

net = Net()
```

---

### 🔹 Loss Functions
```python
criterion = nn.CrossEntropyLoss()
loss = criterion(output, target)
```

---

### 🔹 Optimizers
```python
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
optimizer.zero_grad()       # reset gradients
loss.backward()             # backprop
optimizer.step()            # update weights
```

---

### 🔹 Data Loading
```python
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
for data, target in train_loader:
    output = net(data)
```

---

### 🔹 Saving & Loading Models
```python
torch.save(net.state_dict(), "model.pth")       # save
net.load_state_dict(torch.load("model.pth"))    # load
net.eval()                                      # inference mode
```

---

### 🔹 TorchScript & JIT
```python
traced_model = torch.jit.trace(net, torch.rand(1, 10))
```

---

## 📚 Sources
- Official [PyTorch Cheat Sheet](https://docs-preview.pytorch.org/pytorch/tutorials/2482/beginner/ptcheat.html)  
- [Sling Academy PyTorch Cheat Sheet](https://www.slingacademy.com/article/pytorch-complete-cheat-sheet/)

---
