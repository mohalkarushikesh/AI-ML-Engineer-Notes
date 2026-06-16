````markdown
# Helpers to Get Learning Rate Right in Deep Learning

Choosing the correct learning rate is **critical** for training stability and performance. Too high → divergence, too low → slow convergence.

Here are the most important **helpers, techniques, and tools** used in practice:

---

## 1. Learning Rate Schedulers

Schedulers automatically adjust the learning rate during training.

### 🔹 Step Decay
- Reduce LR after fixed intervals (epochs)
```text
lr = lr * decay_rate
````

***

### 🔹 Exponential Decay

* Smooth continuous decay

```text
lr = lr * e^(-kt)
```

***

### 🔹 Cosine Annealing

* Smooth periodic decay
* Often used with restarts (SGDR)

***

### 🔹 Reduce on Plateau

* Reduce LR when validation loss stops improving
* Very practical and widely used

***

## 2. Learning Rate Finder

A powerful technique to **experimentally find the best LR**.

### Idea:

* Gradually increase LR from very small → large
* Track loss behavior

### Result:

* Pick LR just before loss starts increasing sharply

👉 Popularized by **Leslie Smith**

***

## 3. Cyclical Learning Rates (CLR)

Instead of decreasing LR, it **cycles between min and max values**.

### Benefits:

* Avoids local minima
* Faster convergence

Example cycle:

```text
lr: 0.0001 → 0.01 → 0.0001
```

***

## 4. Warm-up Strategy

* Start with a **very small LR**
* Gradually increase at the beginning

### Why?

* Stabilizes early training
* Prevents exploding gradients

***

## 5. Adaptive Optimizers

Automatically adjust learning rates per parameter.

### Common Methods:

* **Adam**
* **RMSProp**
* **Adagrad**

### Advantage:

* Less manual tuning required

***

## 6. Batch Size Scaling Rule

Learning rate is often tied to batch size.

### Rule of Thumb:

```text
New LR = Base LR × (New Batch Size / Base Batch Size)
```

***

## 7. Gradient Monitoring

Watch gradients during training:

* Exploding gradients → LR too high
* Vanishing gradients → LR too low

***

## 8. Loss Curve Inspection

Plot training loss:

| Behavior           | Meaning     |
| ------------------ | ----------- |
| Oscillating wildly | LR too high |
| Very slow decrease | LR too low  |
| Smooth convergence | LR is good  |

***

## 9. Hyperparameter Tuning

Use:

* Grid Search
* Random Search
* Bayesian Optimization

To find the optimal learning rate.

***

## 10. Default Starting Points

Good initial LRs:

| Optimizer | Default LR |
| --------- | ---------- |
| SGD       | 0.01       |
| Adam      | 0.001      |
| RMSProp   | 0.001      |

***

## ✅ Best Practice Recipe

A strong real-world approach:

1. Use **Adam optimizer**
2. Run a **learning rate finder**
3. Apply **LR scheduler (ReduceLROnPlateau or Cosine)**
4. Optionally add **warm-up**
5. Monitor loss and adjust

***

## 🔚 Summary

Learning rate tuning is made easier using:

* LR schedulers
* Learning rate finder
* Cyclical learning
* Warm-up strategies
* Adaptive optimizers

👉 These tools ensure faster, stable, and more effective model training.

***

```
```
