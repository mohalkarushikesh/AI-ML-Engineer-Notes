# **AI & Machine Learning (AI/ML) In-Depth Cheatsheet**  
*(Comprehensive, Structured, Mnemonic-Free, All Topics Covered)*

---

## **1. AI & ML Fundamentals**
| Concept | Definition | Key Points |
|--------|------------|----------|
| **Artificial Intelligence (AI)** | Machines simulating human intelligence | Includes reasoning, learning, perception, language |
| **Machine Learning (ML)** | Subset of AI; learns from data without explicit programming | 3 Types: Supervised, Unsupervised, Reinforcement |
| **Deep Learning (DL)** | Subset of ML using neural nets with many layers | Requires large data + GPUs |
| **Data Science** | Interdisciplinary field using stats, ML, domain knowledge | Focus: Extract insights from data |
| **Narrow AI** | Task-specific (e.g., image recognition) | Most current AI |
| **General AI (AGI)** | Human-level intelligence across tasks | Not yet achieved |
| **Superintelligence** | Surpasses human intelligence | Theoretical |

---

## **2. Types of Machine Learning**

### **A. Supervised Learning** *(Labeled Data)*
| Model | Use Case | Algorithm Type |
|------|---------|----------------|
| Linear Regression | Predict continuous value | Regression |
| Logistic Regression | Binary/Multi-class classification | Classification |
| Decision Trees | Classification & Regression | Tree-based |
| Random Forest | Robust ensemble | Bagging |
| Gradient Boosting (XGBoost, LightGBM, CatBoost) | High accuracy | Boosting |
| Support Vector Machines (SVM) | Classification (linear/non-linear) | Margin-based |
| K-Nearest Neighbors (KNN) | Instance-based classification/regression | Lazy learner |
| Naive Bayes | Text classification (e.g., spam) | Probabilistic |
| Neural Networks (MLP) | Complex patterns | Feedforward |

---

### **B. Unsupervised Learning** *(No Labels)*
| Technique | Purpose |
|---------|--------|
| K-Means | Clustering (centroid-based) |
| Hierarchical Clustering | Dendrogram-based grouping |
| DBSCAN | Density-based clustering |
| Gaussian Mixture Models (GMM) | Probabilistic clustering |
| Principal Component Analysis (PCA) | Dimensionality reduction |
| t-SNE / UMAP | Visualization (non-linear) |
| Autoencoders | Feature learning / compression |
| Apriori / FP-Growth | Association rule mining |

---

### **C. Semi-Supervised Learning**
| Method | Description |
|-------|-------------|
| Self-Training | Model labels high-confidence data |
| Co-Training | Two models train on different views |
| Label Propagation | Graph-based label spreading |
| Generative Models (VAE + pseudo-labels) | Use generative models to infer labels |

---

### **D. Reinforcement Learning (RL)**
| Algorithm | Type | Key Idea |
|---------|------|---------|
| Q-Learning | Off-policy | Learn value of state-action |
| SARSA | On-policy | Update based on actual action |
| DQN | Value-based | Neural net for Q-values |
| Policy Gradient (REINFORCE) | Policy-based | Direct policy optimization |
| Actor-Critic (A2C/A3C) | Hybrid | Actor (policy) + Critic (value) |
| PPO | On-policy | Clipped objective for stability |
| DDPG | Off-policy (continuous) | Deterministic policy gradient |
| TD3 | Off-policy | Twin critics + delayed updates |

---

## **3. Neural Networks & Deep Learning**

### **A. Core Architectures**
| Architecture | Best For | Key Components |
|-------------|---------|----------------|
| **MLP (Feedforward)** | Tabular data | Dense layers |
| **CNN** | Images, grids | Conv, Pooling, BatchNorm |
| **RNN** | Sequences | Hidden state, Feedback loop |
| **LSTM / GRU** | Long sequences | Gates (forget, input, output) |
| **Transformer** | NLP, sequences | Self-Attention, Multi-Head |
| **GAN** | Generation | Generator + Discriminator |
| **Autoencoder** | Compression | Encoder → Decoder |
| **Diffusion Models** | Image generation | Noise → Denoise |
| **Graph Neural Nets (GNN)** | Graphs | Message passing |

---

### **B. Activation Functions**
| Function | Formula | Range | Pros | Cons |
|--------|--------|-------|------|------|
| **ReLU** | `f(x) = max(0, x)` | [0, ∞) | Fast, avoids vanishing grad | Dying ReLU |
| **Leaky ReLU** | `f(x) = max(αx, x)` | (-∞, ∞) | Fixes dying ReLU | α tuning |
| **ELU** | `α(e^x - 1)` if x<0 | (-α, ∞) | Smooth, negative values | Slower |
| **Swish** | `x * sigmoid(x)` | (-∞, ∞) | Smooth, non-monotonic | Compute cost |
| **GELU** | `x * Φ(x)` | (-∞, ∞) | Used in BERT, smooth | Complex |
| **Sigmoid** | `1/(1+e^(-x))` | (0,1) | Probabilistic output | Vanishing grad |
| **Tanh** | `(e^x - e^(-x))/(e^x + e^(-x))` | (-1,1) | Zero-centered | Vanishing grad |
| **Softmax** | `e^{x_i} / Σ e^{x_j}` | [0,1] | Multi-class prob | Not for single output |

---

## **4. Loss Functions**

| Loss | Formula | Use Case |
|------|--------|---------|
| **MSE** | `(1/n)Σ(y - ŷ)²` | Regression |
| **MAE** | `(1/n)Σ\|y - ŷ\|` | Robust regression |
| **Huber** | `0.5(y-ŷ)²` if \|diff\|<δ else `δ\|diff\| - 0.5δ²` | Robust to outliers |
| **Cross-Entropy** | `-Σ y log(ŷ)` | Multi-class |
| **Binary Cross-Entropy** | `- [y log(ŷ) + (1-y) log(1-ŷ)]` | Binary |
| **Hinge** | `max(0, 1 - y*ŷ)` | SVM |
| **Focal Loss** | `-α(1-ŷ)^γ log(ŷ)` | Imbalanced classes |
| **KL Divergence** | `Σ p log(p/q)` | Distribution matching |
| **Contrastive Loss** | For Siamese nets | Similarity learning |

---

## **5. Optimizers**

| Optimizer | Update Rule | Key Features |
|---------|------------|--------------|
| **SGD** | `θ = θ - η∇L` | Simple, momentum option |
| **SGD + Momentum** | `v = βv + ∇L; θ -= ηv` | Accelerates convergence |
| **Nesterov** | Look-ahead momentum | Better than standard |
| **Adagrad** | Adaptive per-param | Good for sparse data |
| **RMSprop** | `E[g²] = βE[g²] + (1-β)g²; θ -= ηg/√E[g²]` | Handles non-stationary |
| **Adam** | Adaptive moments (1st & 2nd) | Default choice |
| **AdamW** | Decoupled weight decay | Better generalization |
| **Nadam** | Nesterov + Adam | Faster convergence |
| **Adadelta** | No learning rate | Adaptive |
| **Lion** | Sign-based, memory efficient | Emerging |

---

## **6. Regularization Techniques**

| Technique | Purpose | How |
|---------|--------|-----|
| **L1 (Lasso)** | Sparsity | Add `λΣ|w|` |
| **L2 (Ridge)** | Shrink weights | Add `λΣw²` |
| **Elastic Net** | L1 + L2 | `λ1|w| + λ2w²` |
| **Dropout** | Prevent co-adaptation | Randomly drop neurons |
| **DropConnect** | Drop weights | More general |
| **Early Stopping** | Halt on val loss plateau | Monitor patience |
| **Data Augmentation** | Increase diversity | Flip, rotate, noise |
| **Batch Normalization** | Stabilize training | Normalize per batch |
| **Layer Normalization** | For RNNs | Normalize per layer |
| **Weight Decay** | Penalize large weights | In optimizer |

---

## **7. Data Processing Pipeline**

```text
Raw Data 
  → Cleaning (missing, duplicates)
  → Encoding (One-Hot, Label, Target)
  → Scaling (MinMax, Standard, Robust)
  → Feature Engineering (poly, log, binning)
  → Splitting (train/val/test)
  → Augmentation (SMOTE, GANs)
  → Pipeline (sklearn/Pandas)
```

| Step | Tools |
|------|-------|
| Cleaning | Pandas, Missingno |
| Encoding | `OneHotEncoder`, `LabelEncoder` |
| Scaling | `StandardScaler`, `MinMaxScaler` |
| Imputation | Mean, Median, KNN, Iterative |
| Outliers | IQR, Z-score, Isolation Forest |

---

## **8. Model Evaluation**

### **Classification Metrics**
| Metric | Formula | Best When |
|-------|--------|----------|
| Accuracy | (TP+TN)/Total | Balanced |
| Precision | TP/(TP+FP) | Minimize FP |
| Recall | TP/(TP+FN) | Minimize FN |
| F1 | 2PR/(P+R) | Imbalanced |
| ROC-AUC | Area under TPR vs FPR | Threshold-independent |
| PR-AUC | Precision-Recall curve | Highly imbalanced |
| Log Loss | -Σ y log(ŷ) | Probabilistic |

### **Regression Metrics**
| Metric | Formula |
|-------|--------|
| MSE | Σ(y-ŷ)²/n |
| RMSE | √MSE |
| MAE | Σ\|y-ŷ\|/n |
| R² | 1 - (SS_res/SS_tot) |
| MAPE | 100/n Σ \|(y-ŷ)/y\| |

### **Validation Strategies**
- Hold-Out
- K-Fold CV
- Stratified K-Fold
- Time Series Split
- Nested CV (hyperparam tuning)

---

## **9. Feature Selection & Engineering**

| Method | Type |
|-------|------|
| Filter (Correlation, Chi²) | Statistical |
| Wrapper (RFE, Forward/Backward) | Search-based |
| Embedded (Lasso, Tree importance) | Model-based |
| Boruta, SHAP | Advanced |

**Engineering Ideas**:  
- Polynomial features  
- Interaction terms  
- Binning / Discretization  
- Log / sqrt transforms  
- Date-time features  
- Text: TF-IDF, embeddings

---

## **10. Ensemble Methods**

| Method | How | Examples |
|-------|-----|---------|
| **Bagging** | Parallel, reduce variance | Random Forest |
| **Boosting** | Sequential, reduce bias | AdaBoost, GBM, XGBoost |
| **Stacking** | Meta-learner on base models | Stacked Generalization |
| **Voting** | Majority (class) / Avg (reg) | Hard/Soft Voting |

---

## **11. Hyperparameter Tuning**

| Method | Pros | Cons |
|-------|------|------|
| Grid Search | Exhaustive | Slow |
| Random Search | Faster, effective | No guarantee |
| Bayesian (Optuna, Hyperopt) | Sample efficiently | Complex |
| Genetic Algorithms | Global search | Slow |
| BOHB | Combines Bayesian + Bandit | Advanced |

---

## **12. MLOps & Deployment**

| Component | Tools |
|--------|-------|
| Experiment Tracking | MLflow, Weights & Biases, Neptune |
| Model Registry | MLflow, Sagemaker |
| Serving | TensorFlow Serving, TorchServe, FastAPI |
| Monitoring | Prometheus, Grafana, Evidently AI |
| CI/CD | GitHub Actions, Jenkins |
| Orchestration | Kubeflow, Airflow |

---

## **13. Advanced Topics**

| Topic | Key Idea |
|------|---------|
| **Transfer Learning** | Pre-train → Fine-tune |
| **Few-Shot Learning** | Learn from 1–5 examples |
| **Meta-Learning** | "Learn to learn" |
| **Self-Supervised Learning** | Labels from data (e.g., BERT pretraining) |
| **Federated Learning** | Train across devices, keep data local |
| **Explainable AI (XAI)** | SHAP, LIME, Integrated Gradients |
| **AutoML** | NAS, HPO, pipeline search |
| **Continual Learning** | Avoid catastrophic forgetting |

---

## **14. Math & Stats Essentials**

| Concept | Formula / Note |
|--------|----------------|
| **Gradient** | ∂L/∂w |
| **Chain Rule** | dz/dx = dz/dy * dy/dx |
| **Hessian** | Second derivatives |
| **Bias-Variance** | Error = Bias² + Variance + Noise |
| **Bayes Theorem** | P(A\|B) = P(B\|A)P(A)/P(B) |
| **Entropy** | -Σ p log p |
| **Cross-Entropy** | -Σ p log q |
| **KL Divergence** | Σ p log(p/q) |

---

## **15. Tools & Frameworks**

| Category | Tools |
|--------|-------|
| **Languages** | Python, R, Julia |
| **Libraries** | scikit-learn, TensorFlow, PyTorch, JAX, XGBoost |
| **Visualization** | Matplotlib, Seaborn, Plotly, Yellowbrick |
| **Big Data** | Spark MLlib, Dask |
| **Cloud** | AWS SageMaker, GCP Vertex, Azure ML |

---

## **Quick Reference Formulas**

```python
# Accuracy
acc = (TP + TN) / (TP + TN + FP + FN)

# F1
f1 = 2 * (precision * recall) / (precision + recall)

# R²
r2 = 1 - (SS_res / SS_tot)

# Adam Update
m = β1*m + (1-β1)*g
v = β2*v + (1-β2)*g²
θ -= α * m̂ / (√v̂ + ε)
```

---

**Legend**:  
- **TP** = True Positive, **TN** = True Negative  
- **FP** = False Positive, **FN** = False Negative  
- **η** = learning rate, **λ** = regularization  
- **α, β** = hyperparameters

---

Backpropagation - efficiently calculating how much each weight and bias contributes to the network's error by propagating the error backward from the output to the input, allowing for systematic adjustments via gradient descent to improve prediction accuracy. It uses the chain rule of calculus to compute gradients layer by layer, making deep learning scalable and effective for complex pattern recognition. 

Gradient descent is the process of finding the minimum of a function (the loss function) by iteratively taking steps in the direction of the steepest descent (the negative of the gradient).

--- 
**Print, Bookmark, Conquer!**  
This cheatsheet is designed for interviews, exams, and daily reference. No fluff. All signal.

## ⚡ Steps for text preprocessing** in NLP 🧹 
- **Lowercasing** → unify text.  
- **Tokenization** → split into words/subwords.  
- **Stopword removal** → drop common filler words.  
- **Stemming/Lemmatization** → reduce to root form.  
- **Punctuation & special chars removal** → clean text.  
- **Handling numbers** → keep/remove/normalize.  
- **Normalization** → fix accents, spacing.  
- **Vectorization** → convert to embeddings/TF‑IDF.  

## 📚 word embeddings in NLP 🚀 Common Methods
- **Word2Vec** → Skip‑gram, CBOW.  
- **GloVe** → Global co‑occurrence + matrix factorization.  
- **FastText** → Subword info (handles OOV words).  
- **ELMo** → Contextual embeddings (bi‑LSTM).  
- **BERT/Transformers** → Deep contextual, bidirectional.  

# 📚 AI & ML Cheat Sheets Collection

- [RAG, LLM, and AI Agent Cheat Sheets](https://blog.dailydoseofds.com/p/9-rag-llm-and-ai-agent-cheat-sheets)
- [AI-ML Cheat Sheets (GitHub)](https://github.com/SamBelkacem/AI-ML-cheatsheets)
- [Data Science & ML Cheat Sheets](https://blog.dailydoseofds.com/p/15-dsml-cheat-sheets)
- [Neural Networks, ML, DL, Big Data Cheat Sheets](https://becominghuman.ai/cheat-sheets-for-ai-neural-networks-machine-learning-deep-learning-big-data-678c51b4b463?gi=9474d1c5bc4f)
- [LLM Cheat Sheet (GitHub)](https://github.com/Abonia1/CheatSheet-LLM)
- [Kaggle Discussion: Cheat Sheets](https://www.kaggle.com/discussions/general/575880)
- [Datacamp Machine Learning Cheat Sheet](https://www.datacamp.com/cheat-sheet/machine-learning-cheat-sheet)
- [GeeksforGeeks ML Algorithms Cheat Sheet](https://www.geeksforgeeks.org/machine-learning/machine-learning-algorithms-cheat-sheet/)
- [Stanford CS229 ML Tips & Tricks Cheat Sheet](https://stanford.edu/~shervine/teaching/cs-229/cheatsheet-machine-learning-tips-and-tricks)
- [Machine Learning Interview Topics Cheat Sheets](https://medium.com/swlh/cheat-sheets-for-machine-learning-interview-topics-51c2bc2bab4f)

