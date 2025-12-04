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

<img width="451" height="450" alt="image" src="https://github.com/user-attachments/assets/8b5a78f0-b0d5-4667-b81e-346adb338839" />

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


Noise in machine learning is random, irrelevant, or inaccurate information in a dataset that obscures the true underlying patterns

---

**Gibbs sampling is a Markov Chain Monte Carlo (MCMC) algorithm used to generate samples from complex, high-dimensional probability distributions by iteratively sampling from conditional distributions.** It is especially useful in Bayesian statistics and machine learning when direct sampling from the joint distribution is difficult.

---

## 🔑 Key Concepts of Gibbs Sampling
- **MCMC Framework**: Gibbs sampling is a type of MCMC method that constructs a Markov chain whose stationary distribution is the target distribution.
- **Conditional Sampling**: Instead of sampling directly from the joint distribution \(p(x_1, x_2, \dots, x_n)\), Gibbs sampling samples each variable sequentially from its conditional distribution given the others.
- **Iterative Updates**: Each iteration updates one variable at a time, cycling through all variables repeatedly.

---

## 📘 Algorithm Steps
1. **Initialization**  
   - Start with an initial guess for all variables \((x_1, x_2, \dots, x_n)\).

2. **Iterative Sampling**  
   - For each variable \(x_i\), sample from its conditional distribution:

$$
     x_i^{(t+1)} \sim p(x_i \mid x_1^{(t+1)}, \dots, x_{i-1}^{(t+1)}, x_{i+1}^{(t)}, \dots, x_n^{(t)})
$$
   - Repeat for all variables in sequence.

4. **Convergence**  
   - After many iterations, the samples approximate the target joint distribution.

---

## 📊 Example
Suppose we want to sample from a bivariate distribution $(p(x, y))$.  
- Step 1: Initialize $x_0, y_0)$.  
- Step 2: Sample $x_{t+1} \sim p(x \mid y_t)$.  
- Step 3: Sample $y_{t+1} \sim p(y \mid x_{t+1})$.  
- Step 4: Repeat until convergence.

---

## ⚙️ Applications
- **Bayesian Inference**: Estimating posterior distributions when closed-form solutions are unavailable.
- **Latent Variable Models**: Widely used in topic models (e.g., Latent Dirichlet Allocation).
- **Image Processing**: Sampling pixel intensities in Markov Random Fields.
- **Econometrics & Genetics**: Handling complex hierarchical models.

---

## ✅ Advantages
- Works well with **high-dimensional distributions**.
- Requires only **conditional distributions**, which are often easier to compute.
- Converges to the correct distribution under mild conditions.

## ⚠️ Limitations
- **Slow mixing**: Convergence can be slow if variables are highly correlated.
- **Burn-in period**: Initial samples may not represent the target distribution and must be discarded.
- **Requires tractable conditionals**: If conditional distributions are hard to compute, Gibbs sampling is impractical.

---

## 📝 Summary
Gibbs sampling is a **powerful MCMC technique** that simplifies sampling by breaking down a joint distribution into conditional distributions. By iteratively updating each variable, it generates samples that approximate the target distribution, making it indispensable in Bayesian statistics, machine learning, and scientific research.

--- 

#### 🧊 Simulated Annealing
- Gradually reduce a “temperature” parameter to help the system settle into low-energy states.
- Useful for escaping local minima during optimization.
 
________________________________________

**Simulated Annealing (SA) is a probabilistic optimization technique inspired by the physical process of annealing in metallurgy. It is used to find near-optimal solutions in large, complex search spaces, especially when the problem contains many local optima.**

---

## 🔍 Core Idea
Simulated Annealing mimics the cooling of metals:
- **High temperature** allows atoms to move freely.
- **Slow cooling** lets atoms settle into a low-energy (optimal) configuration.
In optimization, this translates to:
- Exploring the solution space widely at first.
- Gradually narrowing the search to settle into a global optimum.

---

## 🧠 Algorithm Steps
1. **Initialize**:
   - Start with a random solution $S$.
   - Set an initial temperature $T$.

2. **Iterative Process**:
   - Generate a neighboring solution  $S'$.
   - Calculate the change in cost $ \Delta E = E(S') - E(S) $.
   - If $\Delta E < 0 $, accept $S'$ (better solution).
   - If $\Delta E > 0$, accept $S'$ with probability:
$$
P = \exp\left(-\frac{\Delta E}{T}\right)
$$
   - Reduce the temperature $T$ using a cooling schedule.

3. **Repeat** until the system is “frozen” (temperature is low or max iterations reached).

---

## 🌡️ Cooling Schedule
The cooling schedule controls how temperature decreases:
- **Linear**: $T_{k+1} = T_k - \alpha$
- **Exponential**: $T_{k+1} = T_k \cdot \alpha$
- **Logarithmic**: $T_k = T_0 / \log(k + 1)$

Choosing the right schedule is crucial for balancing exploration and convergence.

---

## 📊 Example: Traveling Salesman Problem (TSP)
- **Goal**: Find the shortest route visiting all cities once.
- **Initial solution**: Random city order.
- **Neighboring solution**: Swap two cities.
- **Cost function**: Total distance.
- SA helps escape local optima by occasionally accepting worse routes early on.

---

## ✅ Advantages
- **Escapes local optima** due to probabilistic acceptance.
- **Simple and flexible** for various problem types.
- **Works well** with discrete and continuous optimization.

## ⚠️ Limitations
- **Sensitive to parameters** like initial temperature and cooling rate.
- **Slow convergence** if cooling is too gradual.
- **No guarantee** of finding the global optimum.

---

## 🧪 Applications
- **Combinatorial optimization**: TSP, scheduling, assignment problems.
- **Machine learning**: Hyperparameter tuning.
- **Image processing**: Restoration and segmentation.
- **Operations research**: Resource allocation, routing.

---

## 📘 Summary
Simulated Annealing is a powerful metaheuristic that balances exploration and exploitation by mimicking thermal cooling. It’s especially useful when the solution space is rugged and traditional methods get stuck in local optima.

Sources: [GeeksforGeeks](https://www.geeksforgeeks.org/artificial-intelligence/what-is-simulated-annealing/), [Baeldung](https://www.baeldung.com/cs/simulated-annealing), [IIT Madras PDF](https://www.cse.iitm.ac.in/~vplab/courses/optimization/SA_SEL_SLIDES.pdf)


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
  
---

- Additional Learning : **A tutorial on energy based learning** by Yann Lecun et al. (2006) [Paper Link](https://web.stanford.edu/class/cs379c/archive/2012/suggested_reading_list/documents/LeCunetal06.pdf)

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

<img width="857" height="235" alt="image" src="https://github.com/user-attachments/assets/b1453911-dfd3-4f21-997f-51fc583c6d3f" />

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
