import numpy as np 
import torch
import torch.nn as nn 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from torchinfo import summary 

torch.manual_seed(42)
np.random.seed(42)

digits = load_digits()
X = digits.data.astype(np.float32)
y = digits.target.astype(np.int64)
    
X /= 16.0 

n_features = X.shape[1]
n_classes = len(np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y              # keep the same class proportions
)     

X_train_t = torch.from_numpy(X_train)
X_test_t = torch.from_numpy(X_test)
y_train_t = torch.from_numpy(y_train)
y_test_t = torch.from_numpy(y_test)

# 64 -> 128 -> 64 -> 10 

model = nn.Sequential(
    nn.Linear(n_features,128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, n_classes)   
)

batch_size = 32
# summary(model, input_size=(batch_size, 64))

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

n_epochs = 300
loss_history = []

for epoch in range(n_epochs):
    model.train()

    optimizer.zero_grad()
    logits = model(X_train_t)
    loss = loss_fn(logits, y_train_t)
    loss.backward()
    optimizer.step()
    
    loss_history.append(loss.item())
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch: {epoch + 1:3d}/{n_epochs}   |   Traning Loss: {loss.item():.4f}")
        
model.eval()
with torch.no_grad():
    test_logits = model(X_test_t)
    predictions = test_logits.argmax(dim=1)
    accuracy = (predictions == y_test_t).float().mean().item()
    
print(f"Test Accuracy: {accuracy * 100:.2f}% ({int(accuracy * len(y_test))}/{len(y_test)} correct)")

plt.figure(figsize=(8, 5))
plt.plot(range(1, n_epochs + 1), loss_history, color="#2563eb")
plt.title("Traning loss over epochs")
plt.xlabel("Epoch")
plt.ylabel("Cross-entropy Loss")
plt.grid(True, alpha=0.3)
plt.tight_layout()

out_path = "loss_curve.png"
plt.savefig(out_path, dpi=120)
print(f"Loss curve saved to path: {out_path}")

n_samples = 8
sample_idx = np.random.choice(len(X_test_t), size=n_samples, replace=False)

model.eval()
with torch.no_grad():
    sample_logits = model(X_test_t[sample_idx])
    sample_probs = torch.softmax(sample_logits, dim=1)
    sample_preds = sample_probs.argmax(dim=1)

print("\nSample predictions:")
for i, idx in enumerate(sample_idx):
    pred = sample_preds[i].item()
    true = int(y_test[idx])
    conf = sample_probs[i, pred].item()
    mark = "OK" if pred == true else "XX"
    print(f"  [{mark}] predicted {pred} (conf {conf:5.1%})  |  actual {true}")

fig, axes = plt.subplots(2, 4, figsize=(8, 4))
for i, ax in enumerate(axes.ravel()):
    idx = sample_idx[i]
    ax.imshow(X_test[idx].reshape(8, 8), cmap="gray_r")
    pred = sample_preds[i].item()
    true = int(y_test[idx])
    color = "green" if pred == true else "red"
    ax.set_title(f"pred {pred} / true {true}", color=color, fontsize=9)
    ax.axis("off")
fig.tight_layout()

samples_path = "sample_predictions.png"
fig.savefig(samples_path, dpi=120)
print(f"Sample predictions image saved to: {samples_path}")
