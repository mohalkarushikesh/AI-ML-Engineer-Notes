# AI/ML Interview Revision Guide

A focused revision sheet for ML/DL/GenAI interviews. Each section has **core concepts** to recall on demand, followed by **practice questions** (with brief answers) to self-test. Cover the answer, try to explain it out loud, then check.

---

## PART 1 — CLASSIC MACHINE LEARNING

### 1.1 Core concepts to know cold

**Supervised vs unsupervised vs reinforcement**
- Supervised: labelled data, learn X → y (regression, classification).
- Unsupervised: no labels, find structure (clustering, dimensionality reduction).
- Reinforcement: agent learns via reward signal from environment.

**Bias–variance tradeoff**
- Bias = error from wrong assumptions (underfitting). Variance = error from sensitivity to training data (overfitting).
- Total error ≈ bias² + variance + irreducible noise.
- High bias → more complex model / more features. High variance → regularization, more data, simpler model.

**Regularization**
- L2 (Ridge): penalizes sum of squared weights, shrinks weights smoothly, keeps all features.
- L1 (Lasso): penalizes sum of absolute weights, drives some weights to zero → feature selection.
- Elastic Net: combines L1 + L2.

**Linear regression** — fits `y = wᵀx + b` by minimizing MSE. Assumes linearity, independence, homoscedasticity, normal residuals.

**Logistic regression** — classification via sigmoid `σ(z) = 1/(1+e⁻ᶻ)`, trained with cross-entropy (log loss). Output is a probability. Decision boundary is linear.

**SVM (Support Vector Machine)**
- Finds the maximum-margin hyperplane separating classes.
- Support vectors = points on the margin that define the boundary.
- Kernel trick: maps data to higher dimensions implicitly (RBF, polynomial) to handle non-linear separation.
- C parameter: controls tradeoff between margin width and misclassification (low C = wider margin, more tolerance).

**Decision trees** — recursively split on feature that best reduces impurity (Gini or entropy/information gain). Prone to overfitting; controlled by max depth, min samples per leaf, pruning.

**Ensembles**
- Bagging (e.g. Random Forest): train many trees on bootstrapped samples + random feature subsets, average → reduces variance.
- Boosting (e.g. AdaBoost, Gradient Boosting, XGBoost): train models sequentially, each correcting prior errors → reduces bias.
- Random Forest = bagging + feature randomness.

**k-NN** — lazy learner, classifies by majority vote of k nearest neighbours. Sensitive to scaling and choice of k.

**k-Means** — unsupervised clustering; assign points to nearest centroid, recompute centroids, repeat. Must choose k (elbow method / silhouette).

**PCA** — dimensionality reduction; projects onto directions of maximum variance (eigenvectors of covariance matrix). Linear, unsupervised.

**Naive Bayes** — probabilistic classifier using Bayes' theorem with the "naive" assumption of feature independence. Fast, good for text.

**Evaluation metrics**
- Classification: accuracy, precision (TP/(TP+FP)), recall (TP/(TP+FN)), F1 (harmonic mean of P&R), ROC-AUC, confusion matrix.
- Regression: MSE, RMSE, MAE, R².
- Use precision/recall over accuracy on imbalanced data.

**Cross-validation** — k-fold splits data into k parts, train on k−1, validate on 1, rotate. Gives robust performance estimate.

**Feature scaling** — standardization (mean 0, std 1) or normalization (0–1). Needed for distance-based (k-NN, SVM, k-means) and gradient-based methods; not for trees.

### 1.2 Practice questions

1. **Q: Explain the bias–variance tradeoff and how you'd diagnose which you're facing.**
   A: High train error + high test error → high bias (underfitting). Low train error + high test error → high variance (overfitting). Plot learning curves; a large train/validation gap signals variance.

2. **Q: When would you use L1 over L2 regularization?**
   A: L1 when you want sparsity / automatic feature selection (many irrelevant features). L2 when features are correlated and you want to keep all but shrink them.

3. **Q: Why is accuracy a poor metric for imbalanced datasets?**
   A: A model predicting only the majority class can score high accuracy while missing the minority class entirely. Use precision, recall, F1, or ROC-AUC / PR-AUC instead.

4. **Q: What is the kernel trick?**
   A: A way to compute inner products in a high-dimensional feature space without explicitly transforming data, letting linear algorithms (SVM) learn non-linear boundaries efficiently.

5. **Q: How does Random Forest reduce overfitting compared to a single decision tree?**
   A: It averages many de-correlated trees (via bootstrapping rows + random feature subsets), reducing variance while keeping low bias.

6. **Q: Bagging vs boosting — one line each.**
   A: Bagging trains models in parallel on random subsets and averages (cuts variance). Boosting trains sequentially, each focusing on prior errors (cuts bias).

7. **Q: Precision vs recall — give a scenario favouring each.**
   A: Spam filter → favour precision (don't flag good mail). Cancer screening → favour recall (don't miss a real case).

8. **Q: Why must you scale features for k-NN but not for decision trees?**
   A: k-NN uses distances, so unscaled large-range features dominate. Trees split on thresholds per feature, so scale is irrelevant.

---

## PART 2 — DEEP LEARNING / NEURAL NETWORKS

### 2.1 Core concepts to know cold

**Perceptron / neuron** — computes `output = activation(wᵀx + b)`. Stacking layers → multilayer perceptron (MLP).

**Activation functions**
- Sigmoid: (0,1), saturates → vanishing gradients.
- Tanh: (−1,1), zero-centered but still saturates.
- ReLU: max(0,x), cheap, mitigates vanishing gradient, but "dying ReLU" (neurons stuck at 0).
- Leaky ReLU / GELU / SiLU: variants fixing dying ReLU; GELU common in transformers.
- Softmax: converts logits to a probability distribution (output layer for multiclass).

**Forward + backpropagation**
- Forward pass computes predictions and loss.
- Backprop applies chain rule to compute gradients of loss w.r.t. each weight, propagating error backward.
- Weights updated via gradient descent.

**Loss functions**
- Regression: MSE.
- Binary classification: binary cross-entropy.
- Multiclass: categorical cross-entropy.

**Optimizers**
- SGD: update = −lr × gradient. Add momentum to accelerate.
- Adam: adaptive per-parameter learning rates using running averages of gradient (1st moment) and squared gradient (2nd moment). Most common default.
- RMSProp, AdaGrad: adaptive predecessors.

**Vanishing / exploding gradients** — gradients shrink or blow up through many layers. Fixes: ReLU, careful init (He/Xavier), batch norm, residual connections, gradient clipping.

**Regularization in DL**
- Dropout: randomly zero a fraction of neurons during training → prevents co-adaptation.
- Batch normalization: normalizes layer inputs per mini-batch → faster, more stable training, mild regularization.
- L2 weight decay, early stopping, data augmentation.

**Weight initialization** — Xavier/Glorot for tanh/sigmoid, He for ReLU. Prevents signal from vanishing/exploding at start.

**CNNs (Convolutional Neural Networks)**
- Convolution layers apply learnable filters → detect local patterns (edges → textures → objects).
- Key ideas: parameter sharing, local receptive fields, translation invariance.
- Pooling (max/avg) downsamples, adds invariance.
- Used for images; architectures: LeNet, AlexNet, VGG, ResNet (residual skip connections), EfficientNet.

**RNNs / LSTMs / GRUs**
- RNNs process sequences with a hidden state carrying memory; suffer vanishing gradients over long sequences.
- LSTM: gates (input, forget, output) + cell state → learns long-range dependencies.
- GRU: simpler, fewer gates, similar performance.

**Learning rate** — most important hyperparameter. Too high → diverge; too low → slow. Use schedules (step decay, cosine), warmup, or adaptive optimizers.

**Epoch / batch / iteration** — epoch = one full pass over data; batch = subset processed per update; iteration = one weight update.

### 2.2 Practice questions

1. **Q: Why do we need non-linear activation functions?**
   A: Without them a deep network collapses into a single linear transformation; non-linearity lets it approximate complex functions.

2. **Q: What causes vanishing gradients and how do you fix them?**
   A: Repeated multiplication of small derivatives (e.g. sigmoid) through many layers. Fixes: ReLU/GELU, He init, batch norm, residual/skip connections, LSTMs for sequences.

3. **Q: How does dropout work and why does it help?**
   A: Randomly deactivates neurons each training step, forcing redundancy and preventing co-adaptation → acts like training an ensemble → reduces overfitting. Disabled at inference (weights scaled).

4. **Q: Explain batch normalization.**
   A: Normalizes each layer's inputs (mean 0, var 1) over the mini-batch, then scales/shifts with learnable params. Stabilizes and speeds training, reduces sensitivity to init.

5. **Q: Adam vs SGD — when use which?**
   A: Adam converges fast with little tuning, great default (esp. NLP/transformers). Well-tuned SGD+momentum often generalizes better on vision tasks and is standard for large CNNs.

6. **Q: What makes CNNs suited to images?**
   A: Parameter sharing (same filter across the image) drastically cuts params, local receptive fields capture spatial structure, and pooling gives translation invariance.

7. **Q: Why do ResNets use skip connections?**
   A: They let gradients flow directly through identity paths, easing training of very deep nets and combating vanishing gradients / degradation.

8. **Q: LSTM vs vanilla RNN?**
   A: LSTM adds a gated cell state (forget/input/output gates) that preserves information over long sequences, solving the RNN's vanishing-gradient memory problem.

9. **Q: You're overfitting a neural net. List five things to try.**
   A: More data / augmentation, dropout, L2 weight decay, early stopping, reduce model size, batch norm.

---

## PART 3 — LLMs / GENERATIVE AI

### 3.1 Core concepts to know cold

**Transformer architecture** — the backbone of modern LLMs. Replaces recurrence with **self-attention**, enabling parallel processing of sequences and long-range dependencies. From "Attention Is All You Need" (2017).

**Self-attention** — each token computes Query, Key, Value vectors. Attention weight = softmax(Q·Kᵀ / √dₖ), then weighted sum of V. Lets each token attend to every other token by relevance.
- **Multi-head attention**: run attention in parallel subspaces to capture different relationships.
- √dₖ scaling prevents large dot products from saturating softmax.

**Positional encoding** — since attention is order-agnostic, position info is injected (sinusoidal, learned, or rotary/RoPE).

**Encoder vs decoder**
- Encoder-only (BERT): bidirectional, good for understanding/classification.
- Decoder-only (GPT family): causal/autoregressive, good for generation.
- Encoder–decoder (T5, original transformer): seq2seq tasks like translation.

**Tokenization** — text split into subword tokens (BPE, WordPiece, SentencePiece). Balances vocab size and handling rare/unknown words.

**Training stages**
1. **Pretraining**: self-supervised next-token prediction on massive corpora → general language ability.
2. **Supervised fine-tuning (SFT)**: train on curated instruction/response pairs.
3. **RLHF / preference optimization**: align to human preferences via a reward model (RLHF) or direct methods (DPO). Makes models helpful/harmless.

**Emergent abilities / scaling laws** — performance improves predictably with model size, data, and compute; some capabilities appear only past a scale threshold.

**Context window** — max tokens the model can attend to at once. Longer = more context but quadratic attention cost (mitigated by FlashAttention, sparse/linear attention).

**Prompting techniques**
- Zero-shot: just ask.
- Few-shot: include examples in the prompt.
- Chain-of-thought (CoT): "think step by step" → better reasoning.
- ReAct: interleave reasoning + tool/action calls.

**RAG (Retrieval-Augmented Generation)** — retrieve relevant documents (usually via vector/embedding similarity search) and feed them into the prompt so the model answers from up-to-date/external knowledge. Reduces hallucination and enables domain grounding.

**Embeddings** — dense vector representations of text where semantic similarity ≈ cosine similarity. Power search, RAG, clustering, recommendations.

**Fine-tuning vs prompting vs RAG**
- Prompting: fastest, no training, limited by context.
- RAG: inject knowledge without retraining; great for factual/changing data.
- Fine-tuning: change model behaviour/style/format; needs data + compute.
- **PEFT / LoRA**: parameter-efficient fine-tuning — train small low-rank adapter matrices instead of all weights → cheap, fast.

**Hallucination** — model generates fluent but false content. Mitigations: RAG grounding, better prompting, citations, lower temperature, verification steps.

**Decoding parameters**
- Temperature: higher = more random/creative, lower = more deterministic.
- Top-k / Top-p (nucleus): sample from the most probable tokens only.
- Greedy vs beam search.

**Quantization** — reduce weight precision (FP16 → INT8/INT4) to shrink memory and speed inference with minor quality loss.

**MoE (Mixture of Experts)** — only a subset of "expert" subnetworks activate per token → large capacity at lower compute per token.

**Agents** — LLMs that plan, call tools/APIs, and act in loops to accomplish multi-step tasks.

### 3.2 Practice questions

1. **Q: Explain self-attention in one paragraph.**
   A: Each token is projected into Query, Key, and Value vectors. A token's attention over others is softmax(Q·Kᵀ/√dₖ); those weights combine the Value vectors into a context-aware representation. This lets every token draw information from every other token, capturing long-range dependencies in parallel.

2. **Q: Why did transformers replace RNNs for language?**
   A: Attention processes all tokens in parallel (RNNs are sequential), handles long-range dependencies without vanishing gradients, and scales far better on modern hardware.

3. **Q: What is the difference between BERT and GPT?**
   A: BERT is encoder-only and bidirectional (masked language modelling) → understanding tasks. GPT is decoder-only and autoregressive (next-token prediction) → generation.

4. **Q: What problem does RAG solve, and how?**
   A: LLMs have fixed, possibly outdated knowledge and hallucinate. RAG retrieves relevant external documents via embedding similarity and adds them to the prompt so answers are grounded in real, current sources.

5. **Q: When would you fine-tune instead of using RAG or prompting?**
   A: When you need to change the model's behaviour, style, tone, or output format consistently — not just inject facts. RAG/prompting are better for knowledge; fine-tuning is better for behaviour.

6. **Q: What is LoRA and why is it popular?**
   A: LoRA freezes the base model and trains small low-rank adapter matrices, cutting trainable parameters and memory by orders of magnitude while approaching full fine-tuning quality.

7. **Q: What causes hallucinations and how do you reduce them?**
   A: The model predicts plausible tokens without a grounding-in-truth mechanism. Reduce with RAG, retrieval citations, lower temperature, chain-of-thought/verification, and constrained prompting.

8. **Q: What does temperature control during generation?**
   A: The randomness of sampling — high temperature flattens the probability distribution (more diverse/creative), low temperature sharpens it (more focused/deterministic).

9. **Q: Why divide by √dₖ in the attention score?**
   A: Large dot products push softmax into saturated regions with tiny gradients; scaling by √dₖ keeps variance stable and training healthy.

10. **Q: What is a Mixture of Experts model?**
    A: An architecture where a router activates only a few expert subnetworks per token, giving huge total capacity while keeping per-token compute low.

---

## PART 4 — CROSS-CUTTING / BEHAVIOURAL-TECHNICAL

Common interview curveballs:

- **"Walk me through a project."** Structure: problem → data → approach → model choice + why → evaluation → results → what you'd improve. Quantify impact.
- **"How do you handle imbalanced data?"** Resampling (SMOTE/undersampling), class weights, appropriate metrics (PR-AUC, F1), threshold tuning.
- **"Model works in dev but fails in production — why?"** Data drift, train/serve skew, leakage during training, distribution shift, latency constraints.
- **"How do you detect data leakage?"** Suspiciously high performance, features that wouldn't exist at prediction time, target info bleeding into features.
- **"How do you pick a model?"** Data size, interpretability needs, latency, linear vs non-linear structure, baseline first then iterate.
- **"Explain a concept simply."** They test communication — practice explaining attention, overfitting, or gradient descent to a non-expert.

---

## QUICK-FIRE SELF-TEST (cover answers)

| # | Prompt | Recall |
|---|--------|--------|
| 1 | Precision formula | TP/(TP+FP) |
| 2 | Recall formula | TP/(TP+FN) |
| 3 | Cuts variance / cuts bias | Bagging / Boosting |
| 4 | Fixes vanishing gradient (activation) | ReLU |
| 5 | Attention scaling factor | √dₖ |
| 6 | Sparsity-inducing regularizer | L1 / Lasso |
| 7 | Grounds LLMs in external knowledge | RAG |
| 8 | Efficient fine-tuning method | LoRA / PEFT |
| 9 | Normalizes layer inputs per batch | Batch norm |
| 10 | Higher = more creative output | Temperature |

---

*Tip: For a strong interview, be able to (a) define each concept in one sentence, (b) explain the intuition, and (c) give a tradeoff or when-to-use. Interviewers probe depth by asking "why" repeatedly — prepare two layers deep on your strongest topics.*


---
---

Tier 1 — study hardest (this is where you'll be grilled)

All of Part 2 (Deep Learning): activations, backprop, optimizers, vanishing/exploding gradients, batch norm, dropout, weight init. Be able to explain and reason about tradeoffs.
Architectures in depth: CNNs (ResNet skip connections, why they work), RNN→LSTM→Transformer evolution, and self-attention mechanics from Part 3.
Training practicalities: learning-rate schedules, batch size effects, mixed precision, gradient clipping, handling overfitting.

Tier 2 — very likely, second priority

Deployment / MLOps — this is the ML-engineer-specific part my guide is currently light on (more below).
Model optimization for serving: quantization, pruning, distillation, ONNX/TensorRT, batching, latency vs throughput.
Classic ML (Part 1) as baseline — they'll expect you to know metrics, bias/variance, regularization cold, but won't go deep on SVM kernels.

Tier 3 — good to have, lighter

LLM/GenAI concepts (Part 3): know transformers and attention well (Tier 1), but RAG/LoRA/prompting are "nice to mention" unless the role is GenAI-flavored.
Research-level derivations: only if senior/PhD-track.
