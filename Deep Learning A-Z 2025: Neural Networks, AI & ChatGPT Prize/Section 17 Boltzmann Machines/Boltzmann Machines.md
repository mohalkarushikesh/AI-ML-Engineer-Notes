## ⚡ Energy-Based Models (EBMs)

### 🔍 Overview
- EBMs define a scalar energy for each configuration of variables and learn to assign **low energy to desirable (data) states** and high energy to others.
- They don’t output probabilities directly but infer patterns by minimizing energy functions.

### 🧠 Core Idea
- Energy function $E(x)$ measures compatibility between input $x$ and model parameters.
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
- **Weights**: Each connection $w_{ij}$ has a weight that influences the energy of the system.
- **Biases**: Each unit has a bias term $b_i$ or $c_j$ that affects its activation probability.

---

### 📉 Energy Function Explained

The energy of a configuration $(v, h)$ is given by:

$$
E(v, h) = -\sum_i b_i v_i - \sum_j c_j h_j - \sum_{i,j} v_i w_{ij} h_j
$$

- $v_i$: **Visible unit** – a piece of input data, like a pixel or number.
- $h_j$: **Hidden unit** – a learned feature that helps explain the input.
- $b_i$: **Visible bias** – adjusts the influence of each input unit.
- $c_j$: **Hidden bias** – adjusts the activation tendency of each hidden unit.
- $w_{ij}$: **Weight** – controls how strongly visible unit $v_i$ and hidden unit $h_j$ interact.
- **Lower energy** → means the model finds the input and features a good match.
- The system uses this to judge how well $v$ and $h$ work together.

---

### 🔄 Learning Mechanism

#### 🔥 Boltzmann Distribution
The probability of a configuration is:

$$
P(v, h) = \frac{e^{-E(v, h)}}{Z}
$$

Where:
- $Z = \sum_{v,h} e^{-E(v, h)}$ is the **partition function**, summing over all possible states.
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

## 🔬 In-depth Look at Restricted Boltzmann Machines

### 🧠 Intuition Behind RBMs
RBMs learn to capture patterns in data by modeling the joint probability distribution of inputs (visible units) and latent features (hidden units). Training allows them to discover correlations and useful representations—especially in unsupervised settings.

- The absence of intra-layer connections avoids feedback loops, simplifying the inference and learning processes.
- Hidden units act as "feature detectors"—each unit tries to represent some abstract aspect of the input.

---

### 📐 Mathematics of RBMs

#### **Probability Distribution**
RBMs define a probability over the visible and hidden vectors using the energy function:

$$
P(v,h) = \frac{1}{Z} e^{-E(v,h)}
$$

Where:
- $Z$ is the **partition function**:  
  $$Z = \sum_{v,h} e^{-E(v,h)}$$
- This ensures $P(v,h)$ is a valid probability distribution.

---

#### **Marginal and Conditional Probabilities**
RBMs exploit conditional independence for efficient inference:
- Given $v$, hidden units $h_j$ are conditionally independent:

$$
P(h_j = 1 \mid v) = \sigma \left(c_j + \sum_i v_i w_{ij} \right)
$$

- Similarly, for visible units:

$$
P(v_i = 1 \mid h) = \sigma \left(b_i + \sum_j h_j w_{ij} \right)
$$

- $\sigma(x)$ is the sigmoid activation function.

---

### 🛠️ Training RBMs

RBMs are typically trained using **Contrastive Divergence (CD)**:
- Start with the data sample $v^{(0)}$
- Sample hidden $h^{(0)}$ from $P(h \mid v^{(0)})$
- Reconstruct $v^{(1)}$ from $P(v \mid h^{(0)})$
- Repeat for limited steps (usually 1–k steps)
- Update weights using gradient approximation:

$$
\Delta w_{ij} \propto \langle v_i h_j \rangle_{\text{data}} - \langle v_i h_j \rangle_{\text{model}}
$$

CD dramatically reduces computational cost vs full gradient descent.

---

### 📚 Advanced Concepts & Variants

| Variant | Feature | Use Case |
|--------|---------|----------|
| Deep Belief Networks (DBNs) | Stack multiple RBMs | Pretraining deep architectures |
| Conditional RBMs | Add conditioning variables | Time-series or sequential data |
| Gaussian RBMs | Replace binary visible units | Modeling real-valued data |

---

### 🧭 Use Cases in Practice

- **Collaborative Filtering**: Recommender systems (e.g., Netflix Prize)
- **Dimensionality Reduction**: Like PCA, but more expressive
- **Pretraining Neural Networks**: Jumpstart learning before supervised fine-tuning
- **Image Denoising & Reconstruction**: Using learned latent patterns

---

## 🌀 Contrastive Divergence (CD)

### ⚙️ Training Algorithm for RBM
- Approximates gradient of log-likelihood.
- **Idea:** Run Gibbs sampling from data distribution but only for few steps (often 1), hence **CD-k**.

### 🔄 Steps
1. Sample $h \sim P(h|v)$
2. Reconstruct $v' \sim P(v|h)$
3. Use difference between $v$ and $v'$ to update weights

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
