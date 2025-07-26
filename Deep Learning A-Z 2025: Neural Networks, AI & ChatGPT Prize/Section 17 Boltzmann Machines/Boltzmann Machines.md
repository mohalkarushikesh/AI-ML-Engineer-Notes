## ⚡ Energy-Based Models (EBMs)

### 🔍 Overview
- EBMs define a scalar energy for each configuration of variables and learn to assign **low energy to desirable (data) states** and high energy to others.
- They don’t output probabilities directly but infer patterns by minimizing energy functions.

### 🧠 Core Idea
- Energy function \( E(x) \) measures compatibility between input \( x \) and model parameters.
- Learning involves shaping this energy landscape so that real data resides in low-energy regions.

---

## 🔬 Boltzmann Machines — In-Depth Exploration

### 🧠 Conceptual Foundation
- Boltzmann Machines are **stochastic recurrent neural networks** inspired by statistical mechanics.
- They model complex probability distributions over binary vectors using an **energy-based framework**.
- The system seeks configurations (states) with **minimal energy**, which correspond to high-probability data patterns.

---

### 🏗️ Architecture Details

- **Units**: Each neuron (unit) is binary (on/off) and can be either visible (input/output) or hidden (latent features).
- **Connections**: Every unit is connected to every other unit (fully connected), including hidden-hidden and visible-visible links.
- **Weights**: Each connection \( w_{ij} \) has a weight that influences the energy of the system.
- **Biases**: Each unit has a bias term \( b_i \) or \( c_j \) that affects its activation probability.

---

### 📉 Energy Function Explained

The energy of a configuration \( (v, h) \) is given by:

$$
E(v, h) = -\sum_i b_i v_i - \sum_j c_j h_j - \sum_{i,j} v_i w_{ij} h_j
$$

- **Lower energy** → more probable configuration.
- The system uses this function to evaluate how “compatible” a visible-hidden pair is.

---

### 🔄 Learning Mechanism

#### 🔥 Boltzmann Distribution
The probability of a configuration is:

$$
P(v, h) = \frac{e^{-E(v, h)}}{Z}
$$

Where:
- \( Z = \sum_{v,h} e^{-E(v, h)} \) is the **partition function**, summing over all possible states.
- This makes training computationally expensive due to the exponential number of configurations.

#### 🧮 Training Objective
- Maximize the likelihood of training data under the model.
- Use **gradient ascent** on the log-likelihood:

$$
\frac{\partial \log P(v)}{\partial w_{ij}} = \langle v_i h_j \rangle_{\text{data}} - \langle v_i h_j \rangle_{\text{model}}
$$

- The difference between expectations under the data and model distributions drives learning.

---

### 🧪 Sampling & Inference

#### 🔁 Gibbs Sampling
- Iteratively update each unit based on its neighbors.
- Converges to a stationary distribution representing the learned data.

#### 🧊 Simulated Annealing
- Gradually reduce a “temperature” parameter to help the system settle into low-energy states.
- Useful for escaping local minima during optimization.

---

### ⚠️ Challenges

- **Scalability**: Fully connected architecture leads to exponential growth in complexity.
- **Slow Convergence**: Reaching thermal equilibrium takes many iterations.
- **Partition Function**: Intractable to compute for large networks.
- **Gradient Noise**: Learning signal is noisy due to sampling-based estimation.

---

### 🧭 Use Cases

- Theoretical modeling of cognitive processes
- Feature learning in small-scale datasets
- Foundation for more practical models like RBMs and DBNs
- 

---

- Additional Learning : **A tutorial on energy based learning by Yann Lecun et al. (2006)**

---

## 🚫 Restricted Boltzmann Machines (RBM)

### ✂️ Simplification
- **No intra-layer connections** among visible or hidden units—making them “restricted.”

### 🧮 Structure
- Visible layer \( v \), Hidden layer \( h \), with only inter-layer connections.

### 🚀 Applications
- Feature extraction
- Dimensionality reduction
- Pretraining for deep networks

### 🔍 Energy Function
Same form as BM but simplified:
$$
E(v, h) = -\sum_i b_i v_i - \sum_j c_j h_j - \sum_{i,j} v_i w_{ij} h_j
$$

---

## 🌀 Contrastive Divergence (CD)

### ⚙️ Training Algorithm for RBM
- Approximates gradient of log-likelihood.
- **Idea:** Run Gibbs sampling from data distribution but only for few steps (often 1), hence **CD-k**.

### 🔄 Steps
1. Sample \( h \sim P(h|v) \)
2. Reconstruct \( v' \sim P(v|h) \)
3. Use difference between \( v \) and \( v' \) to update weights

> Very efficient compared to pure maximum likelihood methods.

---

## 🧬 Deep Belief Networks (DBN)

### 🧱 Architecture
- Stack of RBMs—each layer is trained individually using unsupervised learning.

### 🔄 Greedy Layer-wise Training
1. Train first RBM on input data.
2. Use its hidden layer outputs as input to next RBM.
3. Repeat for deeper layers.

### 🎯 Purpose
- Pretraining deep networks
- Capturing hierarchical representations

---

## 🌌 Deep Boltzmann Machines (DBM)

### 🌐 Extension of BM with multiple layers
- Like DBNs but with **undirected** connections between all layers.

### 🧠 Highlights
- Captures complex dependencies in data.
- All layers jointly trained via approximate inference methods (e.g., mean-field approximation).

### ⚠️ Challenges
- Very difficult to train due to deep architecture and complex energy functions.

---

## 🧩 Summary Comparison

| Model                  | Connections           | Training Method        | Use Case                          |
|------------------------|-----------------------|------------------------|------------------------------------|
| Boltzmann Machine      | Fully connected       | MCMC / Gradient        | Theoretical understanding          |
| Restricted BM (RBM)    | Bipartite (no intra)  | Contrastive Divergence | Feature extraction, unsupervised  |
| Deep Belief Network    | Stack of RBMs         | Layer-wise CD          | Pretraining, representation learning |
| Deep Boltzmann Machine | All layers connected  | Approximate inference  | Modeling deep dependencies         |

---
