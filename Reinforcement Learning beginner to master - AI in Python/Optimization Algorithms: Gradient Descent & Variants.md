## 🧠 Optimization Algorithms: Gradient Descent & Variants

### 🔧 Gradient Descent
An iterative optimization algorithm that minimizes a function by repeatedly taking steps in the **opposite direction** of the gradient.

**Update Rule:**
$$
x_{n+1} = x_n - \eta \nabla f(x_n)
$$

Where:
- $\eta$: learning rate
- $\nabla f(x_n)$: gradient of the function at $x_n$

---

### 🧮 Variants of Gradient Descent

- **Batch Gradient Descent**  
  Updates parameters **after computing the gradient over the entire training dataset**.

- **Stochastic Gradient Descent (SGD)**  
  Updates parameters using the gradient from **one randomly selected training example** per iteration.
$$
x_{n+1} = x_n - \eta \nabla f(x_i)
$$

- **Mini-Batch Gradient Descent**  
  Uses a **small, randomly selected subset** (mini-batch) of the training data for each update — a compromise between BGD and SGD.

---

## ♟️ Chessboard of Optimization Concepts

| Chess Piece   | Concept                          | Description |
|---------------|----------------------------------|-------------|
| **Pawn**      | Stochastic Gradient Descent (SGD) | Small, incremental updates using one data point at a time. |
| **Rook**      | Gradient Descent                 | Straightforward updates using the full dataset — steady but slow. |
| **Knight**    | Batch Gradient Descent           | Moves in bursts — uses mini-batches for balance between speed and accuracy. |
| **Bishop**    | Convergence                      | Diagonal and strategic — represents the goal of optimization: reaching a stable minimum. |
| **Queen**     | Adam Optimizer                   | Versatile and powerful — combines momentum and adaptive learning rates. |
| **King**      | Final Loss / Global Minimum      | The ultimate goal — minimizing the loss function effectively. |
| **Castle**    | Momentum                         | Adds inertia to updates — helps escape local minima and smooths learning. |
| **Check**     | Overfitting                      | A warning state — when the model fits training data too closely. |
| **Checkmate** | Divergence                       | Optimization failure — when updates explode or never converge. |

---

## ⚙️ Key Concepts

- **Gradient**: Inclination of the line (slope); direction of steepest ascent.
- **Gradient Descent**: Moves opposite to the gradient to find a **local minimum**.
- **Gradient Ascent**: Moves along the gradient to find a **local maximum**.
- **Loss Function**: Measures how well the model fits the data.
- **Gradient of Loss**:
$$
\text{gradient} = \frac{\partial L}{\partial \theta}
$$
  Where:
  - $L$: loss function
  - $\theta$: model parameters

- **Parameter Update Rule**:
$$
\theta = \theta - \eta \cdot \text{gradient}
$$

---

## ⚡ Comparison: GD vs SGD

| Feature           | Gradient Descent (GD) | Stochastic Gradient Descent (SGD) |
|------------------|------------------------|-----------------------------------|
| **Speed**        | Slower                 | Much faster for large datasets    |
| **Noise Robustness** | Less robust         | More robust to noisy data         |
| **Memory Usage** | High                   | Low                               |
| **Convergence**  | More stable            | Slower but quicker to start       |

---

## 🚀 Why Use SGD?

1. Handles **large datasets** efficiently  
2. Works well with **non-convex optimization problems**  
3. Enables **online learning** — updates with streaming data

---

## 📉 Learning Rate & Batch Size Effects

- **Smaller learning rate** → slower but more stable convergence  
- **Larger learning rate** → faster but risk of divergence  
- **Smaller batch size** → slower but more accurate  
- **Larger batch size** → faster but less precise

---

## 🔄 Definitions

- **Convergence**: Coming together from different directions (merging into one).
- **Divergence**: Branching out from a common point into different directions.
- **Introducing randomness**: Helps the model learn faster and escape local minima.
