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

- CD is a **fast learning algorithm** for Restricted Boltzmann Machines (RBMs).
- It approximates the gradient of the log-likelihood of the data.
- Instead of full MCMC sampling, it uses short Gibbs sampling chains—often **just 1 step**, hence **CD-$k$**.

---

### 🔍 Why Use It?

- Computing the exact gradient requires evaluating the partition function—**very expensive**.
- CD avoids this by:
  - Running a few sampling steps from the data,
  - Comparing the original input and the reconstructed version.
- Even **CD-1** is effective for learning useful features and representations.

---

### 🔄 Steps of CD-$k$

Let $v$ be the visible layer input and $k$ be the number of Gibbs sampling steps:

1. **Positive Phase**:
   - Sample hidden units $h$ from $P(h | v)$ using current weights.
   - This captures statistics from the **data distribution**.

2. **Negative Phase (Reconstruction)**:
   - Reconstruct visible units $v' \sim P(v | h)$.
   - Run $k$ steps of Gibbs sampling:
     - Sample hidden units $h' \sim P(h | v')$,
     - Then sample visible units $v'' \sim P(v | h')$,
     - Repeat $k$ times if $k > 1$.

3. **Weight Update**:
   - Use the difference in expectations between the data and reconstruction to update:
  
$$
\Delta w_{ij} \propto \langle v_i h_j \rangle_{\text{data}} - \langle v_i h_j \rangle_{\text{reconstruction}}
$$

---

<img width="1059" height="376" alt="image" src="https://github.com/user-attachments/assets/bb3efbdd-7fce-4c06-92df-baf4a6d57096" />
<img width="1002" height="367" alt="image" src="https://github.com/user-attachments/assets/01c95081-a07a-4dd9-a0f7-49a6757e9ffe" />
<img width="1047" height="388" alt="image" src="https://github.com/user-attachments/assets/adaef6f4-b508-4aa2-b78b-662436a377e1" />

---

- Additional Learning
  - **A fast learning algorithm for deep belief nets** by Geoffrey Hinton et al. (2006) [Paper Link](https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf)
  - **Notes on Contrastive Divergence** by oliver woodford (2012) [Paper Link](http://www.robots.ox.ac.uk/~ojw/files/NotesOnCD.pdf)
    
---
### 📌 Key Points

- No need to compute the partition function—**makes training scalable**.
- Often used for:    
  - **Unsupervised feature learning**,
  - **Dimensionality reduction**,
  - **Pretraining** layers in deep networks.
- Performance depends on parameters:
  - Choice of $k$, learning rate, initialization,
  - Mini-batch size, and momentum.
- Works well even with binary units, but also extendable to Gaussian and other types.
  
---

## 🧬 Deep Belief Networks (DBN)

### 🧱 Architecture
- DBNs are composed of multiple layers of **Restricted Boltzmann Machines (RBMs)**.
- Each RBM consists of a **visible layer** ($v$) and a **hidden layer** ($h$), with symmetrical connections and no intra-layer connections.
- Layers are stacked such that the hidden layer of one RBM becomes the visible layer of the next.

### 🔄 Greedy Layer-wise Training
1. **Unsupervised Pretraining**:  
   - Train the first RBM on the input data $x$ to learn $P(h_1 | x)$.
   - Use the activations of $h_1$ as input for the second RBM: learn $P(h_2 | h_1)$.
   - Repeat this process for all subsequent layers ($h_3, h_4, \dots$).
2. **Fine-tuning with Supervised Learning**:  
   - Once all RBMs are pretrained, the whole DBN is fine-tuned using **backpropagation** to improve performance on a labeled task (e.g., classification).

### 🎯 Purpose
- **Pretraining** helps in initializing deep networks efficiently, overcoming issues like vanishing gradients.
- DBNs learn **hierarchical feature representations**:  
  - Lower layers capture simple features (edges, shapes).  
  - Higher layers capture abstract patterns (objects, categories).

---
<img width="1084" height="473" alt="image" src="https://github.com/user-attachments/assets/3e8ef3c6-63ef-469f-bd7e-57e93d1382e3" />
---

How DBNs differ from other deep architectures like autoencoders or CNNs? 🧠🔍

---

## 🔍 DBN vs. Autoencoders vs. CNNs

| Feature/Aspect         | 🧬 DBN                                     | 🔄 Autoencoder                              | 🧠 CNN                                          |
|------------------------|-------------------------------------------|---------------------------------------------|------------------------------------------------|
| **Core Idea**          | Stacked RBMs trained layer-wise           | Encode–decode input through bottleneck      | Learn spatial features via convolution filters |
| **Training Type**      | Unsupervised pretraining + supervised fine-tuning | Typically unsupervised                      | Mostly supervised                              |
| **Architecture**       | Multiple RBMs stacked (generative model)  | Symmetric encoder-decoder structure         | Convolution + pooling layers                   |
| **Data Suitability**   | Tabular, structured, binary input         | Any input—images, text, tabular             | Especially effective on image and spatial data |
| **Feature Hierarchy**  | Learns abstract features layer by layer   | Learns compressed representation            | Captures local patterns using receptive fields |
| **Applications**       | Pretraining deep nets, feature discovery  | Dimensionality reduction, anomaly detection | Image classification, object detection         |
| **Latent Space**       | Hierarchical probabilistic representation | Continuous low-dimensional code             | Learned feature maps in spatial hierarchy      |

---

### 🧠 Key Takeaways

- **DBNs** are probabilistic generative models—excellent for discovering deep, abstract features in unlabeled data and initializing deep architectures.
- **Autoencoders** compress input into a latent space and reconstruct it—great for understanding data structure and noise reduction.
- **CNNs** dominate in vision tasks due to their ability to detect spatial hierarchies and patterns across pixels.

---

- Additinal Learning:
  - **Greedy Layer-Wise Training of Deep Networks** by Yoshua Bengio et al. (2006) [Paper Link](http://www.iro.umontreal.ca/~lisa/pointeurs/BengioNips2006All.pdf)
  - **The Wake-sleep algorithm for unsupervised neural networks** by Groffrey Hinton et al. (1995) [Paper Link](http://www.gatsby.ucl.ac.uk/~dayan/papers/hdfn95.pdf)
---

## 🌌 Deep Boltzmann Machines (DBM)

### 🌐 Extension of BM with Multiple Layers

- DBMs are **deep generative models** built from layers of binary units, each connected via **undirected weights**.
- Unlike Deep Belief Networks (DBNs), all connections between layers are **symmetric and bidirectional**—no top-down or bottom-up directional assumption.
- Stacked **Restricted Boltzmann Machines (RBMs)** are used, but training is done **jointly** across all layers instead of greedily.

Let layers be indexed as $v$, $h^{(1)}$, $h^{(2)}$, ..., $h^{(L)}$ where:
- $v$: visible layer (input)
- $h^{(l)}$: hidden layers from layer $1$ to layer $L$

---

<img width="1076" height="479" alt="image" src="https://github.com/user-attachments/assets/35eab926-a331-45e1-9d8c-b83051e974b8" />

---

### 🧠 Highlights

- **Hierarchical latent representation**:
  - Lower hidden layers capture simple correlations.
  - Higher layers model **abstract and global features**.
- **Undirected connections** allow **inter-layer symmetry**, enabling richer interactions.
- Learns a **joint probability distribution** over inputs and hidden features:

$$
P(v, h^{(1)}, h^{(2)}, \dots, h^{(L)}) = \frac{1}{Z} e^{-E(v, h^{(1)}, \dots, h^{(L)})}
$$

- **Energy Function Example** (3-layer DBM):

$$
E(v, h^{(1)}, h^{(2)}) = -v^\top W^{(1)} h^{(1)} - h^{(1)\top} W^{(2)} h^{(2)} - b^\top v - c^{(1)\top} h^{(1)} - c^{(2)\top} h^{(2)}
$$

---

### 🧠 Training with Approximate Inference

- Full inference is **intractable** due to the nested structure.
- DBMs use **variational methods** like:
  - **Mean-field approximation**
  - **Stochastic gradient descent**
  - **Persistent Contrastive Divergence (PCD)**

#### 🔄 Mean-Field Overview

- Replaces sampling with **deterministic updates** of unit states.
- Iteratively estimates expected values $\langle h_j \rangle$ for hidden units.
- Approximates posterior distributions needed for weight updates.

---

### ⚠️ Challenges

- **Training difficulty**:
  - Requires careful initialization, often using pre-trained RBMs.
  - Gradient signals diminish over layers—makes optimization harder.
- **Slow convergence**:
  - Sampling-based methods like MCMC are computationally expensive.
- **Hyperparameter sensitivity**:
  - Small changes in learning rate, weight decay, or layer size affect performance.

---

### 🧪 Applications

- Modeling high-dimensional data distributions.
- Unsupervised learning on complex datasets (images, text).
- Pretraining components of deep neural networks.

---
- Additional Learning : **Deep Boltxmann Machines** by Ruslan Salahutdinov et al. (2009) [Paper Link](http://www.utstat.toronto.edu/~rsalakhu/papers/dbm.pdf)
---

## 🧩 Summary Comparison

| Model                  | Connections           | Training Method        | Use Case                          |
|------------------------|-----------------------|------------------------|------------------------------------|
| Boltzmann Machine      | Fully connected       | MCMC / Gradient        | Theoretical understanding          |
| Restricted BM (RBM)    | Bipartite (no intra)  | Contrastive Divergence | Feature extraction, unsupervised  |
| Deep Belief Network    | Stack of RBMs         | Layer-wise CD          | Pretraining, representation learning |
| Deep Boltzmann Machine | All layers connected  | Approximate inference  | Modeling deep dependencies         |

---
