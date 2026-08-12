Ah, got it — you want *volume*, a big question bank to grind through, not just a few worked examples. Fair. I can't literally give infinite, but here's a large bank of practice problems, and I'll generate **more on demand** any time you say "more" — that's the practical version of unlimited.

These are problem statements only (so they're real exercises). Ask me for the solution to any number and I'll walk it through.

**Legend:** 🟢 beginner · 🟡 medium · 🔴 advanced

---

**NumPy / Tensors**
1. 🟢 Create a 5×5 identity matrix and multiply it by 7.
2. 🟢 Given a 1D array, replace all negative values with 0 (ReLU by hand).
3. 🟢 Compute row-wise and column-wise sums of a 3×4 matrix.
4. 🟡 Normalize a `(N, D)` dataset to zero mean, unit variance per feature.
5. 🟡 Implement one-hot encoding of an integer label array without sklearn.
6. 🟡 Compute pairwise Euclidean distances between rows of two matrices (vectorized, no loops).
7. 🔴 Implement numerically stable log-sum-exp.
8. 🔴 Implement batched matrix multiplication and verify against `np.einsum`.
9. 🔴 Given predictions and labels, compute a confusion matrix using only NumPy.

**Classic ML (sklearn)**
10. 🟢 Train a decision tree on Iris and print its depth.
11. 🟢 Fit a KNN classifier and try k = 1, 5, 15; compare accuracy.
12. 🟡 Build a preprocessing pipeline: impute missing values → scale → logistic regression.
13. 🟡 Use `train_test_split` with stratification on an imbalanced target and verify class ratios.
14. 🟡 Compute precision, recall, F1, and ROC-AUC for a binary classifier.
15. 🔴 Implement k-fold cross-validation manually (no `cross_val_score`).
16. 🔴 Handle class imbalance with `class_weight='balanced'` and compare to SMOTE.
17. 🔴 Extract and plot feature importances from a Random Forest.

**Regression from scratch**
18. 🟢 Implement mean squared error.
19. 🟢 Implement the sigmoid and its derivative.
20. 🟡 Implement closed-form linear regression via the normal equation.
21. 🟡 Implement gradient descent for linear regression and plot loss over epochs.
22. 🔴 Add L2 regularization to your from-scratch linear regression.
23. 🔴 Implement mini-batch gradient descent and compare convergence to full-batch.

**PyTorch core**
24. 🟢 Create a tensor, move it to GPU if available, and print its device.
25. 🟢 Build a 3-layer MLP with ReLU for regression (1 output).
26. 🟢 Compute gradients of `y = x²` at x = 3 using autograd.
27. 🟡 Write a full training loop with train/val split and accuracy tracking.
28. 🟡 Add a learning-rate scheduler (StepLR or cosine) to a training loop.
29. 🟡 Implement gradient clipping in a training loop.
30. 🔴 Write a custom `Dataset` and `DataLoader` for a CSV file.
31. 🔴 Implement mixed-precision training with `torch.cuda.amp`.
32. 🔴 Build a custom loss function as an `nn.Module` and train with it.

**CNNs**
33. 🟢 Compute the output size of a conv layer given kernel/stride/padding by hand.
34. 🟢 Define a conv → BN → ReLU → pool block.
35. 🟡 Build a CNN for CIFAR-10-shaped input and count total parameters.
36. 🟡 Add data augmentation (random flip, crop) with torchvision transforms.
37. 🔴 Implement a residual block and stack 3 of them.
38. 🔴 Load a pretrained ResNet and fine-tune only the final layer (transfer learning).
39. 🔴 Implement global average pooling instead of a flatten + FC head.

**RNN / LSTM**
40. 🟢 Run a vanilla RNN over a sequence and inspect hidden state shape.
41. 🟡 Build an LSTM for many-to-one sentiment classification.
42. 🟡 Build a bidirectional LSTM and explain the output dimension change.
43. 🔴 Handle variable-length sequences with pack/pad.
44. 🔴 Implement a char-level text generator (many-to-many).

**Regularization**
45. 🟢 Add dropout between two linear layers.
46. 🟢 Add weight decay to an optimizer.
47. 🟡 Manually add an L1 penalty to the loss.
48. 🟡 Implement early stopping with a patience counter.
49. 🔴 Implement label smoothing cross-entropy from scratch.
50. 🔴 Implement mixup data augmentation.

**Transformers / Attention**
51. 🟢 Use `nn.MultiheadAttention` and read the output/weight shapes.
52. 🟡 Implement scaled dot-product attention from scratch.
53. 🟡 Add a causal (look-ahead) mask to attention.
54. 🔴 Implement multi-head attention from scratch (split/combine heads).
55. 🔴 Implement sinusoidal positional encoding.
56. 🔴 Build a single Transformer encoder block (attention + FFN + residual + norm).

**Deployment / MLOps**
57. 🟢 Save and load a model's `state_dict` correctly.
58. 🟢 Run batched inference under `torch.no_grad()`.
59. 🟡 Wrap a model in a FastAPI `/predict` endpoint.
60. 🟡 Measure inference latency and throughput for different batch sizes.
61. 🔴 Export a model to ONNX and run it with onnxruntime.
62. 🔴 Apply dynamic quantization and compare model size + speed.
63. 🔴 Sketch a data-drift monitor: compare live feature distributions to training stats.

**System design (verbal — rehearse out loud)**
64. 🔴 Design an image-classification service handling 10k requests/sec.
65. 🔴 Design a training pipeline that retrains daily on new data.
66. 🔴 How would you A/B test a new model version safely in production?
67. 🔴 A deployed model's accuracy drops over two weeks — debug it.

---

That's 67 to start. Say **"more"** for another batch (I can go deeper into any single topic, or add NLP-preprocessing, recommendation systems, or interview-style ML system design). Or say **"solve #N"** and I'll give the full worked solution for that problem.

Which way do you want to go?

Here's batch 2 — another 60, going into topics the first set didn't cover plus harder variants. Same legend: 🟢 beginner · 🟡 medium · 🔴 advanced. Say **"solve #N"** for a worked solution, or **"more"** again.

---

**Data preprocessing / feature engineering**
68. 🟢 Encode categorical columns with label encoding vs one-hot — when each?
69. 🟢 Bin a continuous age column into groups.
70. 🟡 Handle missing values three ways: drop, mean-impute, forward-fill — code each.
71. 🟡 Detect and cap outliers using the IQR method.
72. 🟡 Create polynomial + interaction features with `PolynomialFeatures`.
73. 🔴 Build a `ColumnTransformer` that scales numerics and one-hots categoricals in one pipeline.
74. 🔴 Prevent data leakage: fit the scaler on train only, then transform val/test — show the wrong way and the right way.

**NLP preprocessing**
75. 🟢 Tokenize a sentence and build a word→index vocabulary.
76. 🟡 Implement bag-of-words and TF-IDF with sklearn.
77. 🟡 Pad/truncate a batch of token sequences to fixed length.
78. 🔴 Use a HuggingFace tokenizer and inspect input_ids + attention_mask.
79. 🔴 Compute cosine similarity between two sentence embeddings.

**Optimizers & training dynamics**
80. 🟢 Compare SGD vs Adam on the same model — which converges faster?
81. 🟡 Implement SGD with momentum from scratch (NumPy).
82. 🟡 Plot a learning-rate warmup + cosine decay schedule.
83. 🔴 Implement the Adam update rule from scratch.
84. 🔴 Diagnose a loss curve that plateaus then spikes — list causes and fixes.
85. 🔴 Implement gradient accumulation to simulate a larger batch size.

**Evaluation & metrics**
86. 🟢 Compute accuracy from logits and labels in PyTorch.
87. 🟡 Plot a ROC curve and compute AUC.
88. 🟡 Plot a precision-recall curve for imbalanced data.
89. 🟡 Choose an optimal classification threshold from the PR curve.
90. 🔴 Implement top-k accuracy for multi-class.
91. 🔴 Compute mean Average Precision (mAP) conceptually for object detection.

**CNNs — deeper**
92. 🟡 Visualize feature maps from the first conv layer.
93. 🟡 Implement depthwise-separable convolution and explain the param savings.
94. 🔴 Implement a 1×1 convolution and explain its purpose (channel mixing / bottleneck).
95. 🔴 Build a U-Net-style encoder-decoder with skip connections for segmentation.
96. 🔴 Implement Grad-CAM to see what a CNN "looks at."

**Transformers — deeper**
97. 🟡 Explain and implement layer normalization from scratch.
98. 🟡 Implement the position-wise feed-forward network of a transformer.
99. 🔴 Stack encoder blocks into a full transformer encoder with embeddings + positional encoding.
100. 🔴 Implement KV-caching conceptually for faster autoregressive generation.
101. 🔴 Explain why attention is O(n²) and name two efficient-attention fixes.

**Applied GenAI / LLM usage (light, since it may come up)**
102. 🟢 Call an LLM API and parse the text response.
103. 🟡 Build a simple RAG loop: embed docs → similarity search → stuff context into prompt.
104. 🟡 Implement chunking of a long document for embedding.
105. 🔴 Add a re-ranking step to a retrieval pipeline.
106. 🔴 Design an eval harness to measure hallucination rate on a QA dataset.

**Deployment / MLOps — deeper**
107. 🟢 Pin dependencies and write a minimal Dockerfile for a model service.
108. 🟡 Add input validation + error handling to a prediction endpoint.
109. 🟡 Log predictions and latencies for monitoring.
110. 🔴 Implement request batching (micro-batching) on the server for throughput.
111. 🔴 Set up a canary deployment: route 5% of traffic to a new model.
112. 🔴 Design model versioning + rollback for a serving system.

**Debugging scenarios (verbal — rehearse out loud)**
113. 🟡 Training loss is NaN after a few steps — list the likely causes.
114. 🟡 Model overfits badly — give five concrete fixes in priority order.
115. 🟡 Train accuracy is high but val accuracy is stuck — what's happening?
116. 🔴 Model works on your laptop but is 10× slower in production — debug.
117. 🔴 Predictions are great offline but bad live — name the usual suspect (train/serve skew) and how you'd confirm it.

**System design — deeper (verbal)**
118. 🔴 Design a recommendation system for an e-commerce site.
119. 🔴 Design a real-time fraud-detection pipeline (latency-critical).
120. 🔴 Design the ML side of a "count objects in a video feed" system (your potato example, scaled up).
121. 🔴 How do you serve a model too large for a single GPU?
122. 🔴 Design an offline batch-scoring job for 100M records nightly.

---

That's 122 total across both batches. From here you can:
- **"more"** → batch 3 (I can go into reinforcement learning, time series, GNNs, or pure Python/DSA-for-ML coding rounds)
- **"solve #N"** → full worked solution for any problem
- **"drill me on Topic X"** → I ask one at a time and grade your spoken/typed answers like a real interviewer

What next?

Batch 3 — 60 more, into areas the first two batches skipped: RL, time series, probability/stats, Python/DSA coding rounds, and a few odds and ends. Same legend: 🟢 · 🟡 · 🔴.

---

**Probability & statistics (common in DS/MLE screens)**
123. 🟢 Compute mean, variance, and std of an array by hand (no `.std()`).
124. 🟢 Simulate 10,000 coin flips and estimate P(heads).
125. 🟡 Implement Bayes' theorem for a disease-test problem (given sensitivity/specificity).
126. 🟡 Run a two-sample t-test with scipy and interpret the p-value.
127. 🟡 Explain and simulate the Central Limit Theorem.
128. 🔴 Implement bootstrap confidence intervals for a mean.
129. 🔴 Design an A/B test: sample size, metric, significance, and how you'd call it.

**Time series**
130. 🟢 Plot a time series and its rolling mean.
131. 🟡 Create lag features and a train/test split that respects time order.
132. 🟡 Check stationarity with the ADF test; difference the series if needed.
133. 🔴 Build an LSTM forecaster with a sliding-window dataset.
134. 🔴 Explain why you must never shuffle time-series data before splitting.

**Reinforcement learning (basics)**
135. 🟢 Define the RL loop: state → action → reward → next state.
136. 🟡 Implement tabular Q-learning for a small gridworld.
137. 🟡 Explain and code an epsilon-greedy action selection.
138. 🔴 Implement a basic DQN (network + replay buffer + target network) outline.
139. 🔴 Explain the exploration vs exploitation tradeoff with a concrete example.

**Graph / recommendation (nice-to-have)**
140. 🟡 Build a user-item interaction matrix and compute item-item cosine similarity.
141. 🔴 Implement matrix factorization for collaborative filtering with gradient descent.
142. 🔴 Explain a Graph Neural Network's message-passing step conceptually.

**Python / DSA for ML coding rounds (they DO test this)**
143. 🟢 Reverse a list and a string without built-ins.
144. 🟢 Count word frequencies in a paragraph using a dict.
145. 🟡 Find the top-k frequent elements in an array (heap).
146. 🟡 Implement binary search.
147. 🟡 Merge two sorted lists.
148. 🔴 Implement your own `train_test_split` with a random seed and no sklearn.
149. 🔴 Given a stream of numbers, maintain a running mean and variance (Welford's algorithm).
150. 🔴 Implement a sliding-window maximum in O(n).

**NumPy / vectorization mastery**
151. 🟡 Replace a double for-loop distance computation with broadcasting.
152. 🟡 Implement k-means clustering in pure NumPy.
153. 🔴 Implement PCA from scratch (covariance → eigendecomposition → project).
154. 🔴 Vectorize the forward pass of a 2-layer neural net (no loops).

**PyTorch internals / gotchas**
155. 🟡 Explain the difference between `.detach()`, `.item()`, and `.numpy()`.
156. 🟡 Show why in-place ops can break autograd, and how to avoid it.
157. 🔴 Implement a custom `autograd.Function` with forward + backward.
158. 🔴 Freeze part of a network (`requires_grad=False`) and train only the head.
159. 🔴 Reproduce results exactly: set all seeds and enable deterministic mode.

**Loss functions**
160. 🟢 When to use MSE vs MAE vs Huber loss?
161. 🟡 Implement focal loss for class imbalance.
162. 🔴 Implement contrastive / triplet loss for embeddings.

**Model interpretability**
163. 🟡 Get feature importances from a linear model's coefficients (with scaling caveat).
164. 🔴 Use SHAP to explain a single prediction (conceptually + code outline).

**Grab-bag / conceptual (verbal — rehearse out loud)**
165. 🟢 Explain overfitting to a non-technical stakeholder.
166. 🟡 When would you choose a gradient-boosted tree over a neural net?
167. 🟡 How do you decide train/val/test split ratios?
168. 🔴 Your model is accurate but the business impact is unclear — how do you connect metrics to value?
169. 🔴 Walk through an end-to-end project you'd build for the potato-counting problem, from data collection to deployment and monitoring.

---

That's **169 total** across three batches — a solid grind bank spanning classic ML, deep learning, transformers, deployment, RL, time series, stats, and DSA.

Honestly, at this point you have more than enough breadth. My suggestion: switch from *collecting* questions to *doing* them. Two high-leverage options:

- **"solve #N"** — I give the full worked solution to any problem so you learn the pattern.
- **"drill me"** — I pick questions one at a time, you answer, I grade you like a real interviewer and push on the "why." This is what actually moves the needle before an interview.

Want batch 4 anyway (RL deep-dive, MLOps tooling, or FAANG-style ML system design), or shall we start drilling?

Batch 4 — 55 more, into the areas I flagged: RL deep-dive, MLOps tooling, and FAANG-style ML system design, plus a few advanced DL topics. Legend: 🟢 · 🟡 · 🔴.

---

**Reinforcement learning — deeper**
170. 🟡 Explain the Bellman equation in plain words and write it out.
171. 🟡 Difference between value-based, policy-based, and actor-critic methods.
172. 🔴 Implement the policy gradient (REINFORCE) update rule.
173. 🔴 Explain the role of the target network and replay buffer in DQN — why each exists.
174. 🔴 Explain how PPO stabilizes policy updates (clipped objective).
175. 🔴 Describe how RLHF applies RL to LLM training (reward model + policy).

**Advanced deep learning**
176. 🟡 Explain and implement gradient checkpointing (trade compute for memory).
177. 🟡 Explain knowledge distillation — teacher/student setup.
178. 🔴 Implement a distillation loss (soft targets with temperature + hard-label term).
179. 🔴 Explain the difference between batch norm, layer norm, group norm, and when each is used.
180. 🔴 Explain how mixed-precision (FP16/BF16) training works and what a loss scaler does.
181. 🔴 Explain what LoRA changes in fine-tuning and why it saves memory.
182. 🔴 Explain data parallelism vs model parallelism vs pipeline parallelism.

**Generative models (beyond LLMs)**
183. 🟡 Explain how an autoencoder works and what the bottleneck does.
184. 🔴 Explain the VAE reparameterization trick and why it's needed.
185. 🔴 Explain the GAN generator/discriminator minimax game and mode collapse.
186. 🔴 Explain the core idea of diffusion models (forward noising, reverse denoising).

**MLOps tooling & practice**
187. 🟢 What goes in a `requirements.txt` vs a Dockerfile vs a CI config?
188. 🟡 Track experiments with MLflow (or Weights & Biases) — what do you log?
189. 🟡 Explain a feature store and the problem it solves.
190. 🟡 Version a dataset — why and how (e.g. DVC)?
191. 🔴 Design a CI/CD pipeline for ML: what triggers retraining, testing, and deployment?
192. 🔴 Explain the difference between online and offline (batch) inference with examples.
193. 🔴 How do you detect and respond to data drift vs concept drift in production?
194. 🔴 Design model monitoring: what metrics beyond accuracy do you track live?

**FAANG-style ML system design (verbal — rehearse out loud, use a framework)**
195. 🔴 Design YouTube video recommendations.
196. 🔴 Design a spam/abuse detection system for a messaging app.
197. 🔴 Design search ranking for an e-commerce site.
198. 🔴 Design a "people you may know" friend-recommendation system.
199. 🔴 Design an ETA-prediction system for a ride-hailing app.
200. 🔴 Design an ad click-through-rate (CTR) prediction system.
201. 🔴 Design content moderation for image uploads at scale.
202. 🔴 Design a real-time speech-to-text transcription service.

> **System-design framework to use for all of the above:** clarify requirements & scale → define the ML problem & metrics → data & labels → features → model choice & baseline → training pipeline → serving (latency/throughput) → evaluation (offline + online/A-B) → monitoring & retraining. Narrate in that order and you'll never freeze.

**Scaling & performance**
203. 🟡 How does batch size affect training speed, memory, and generalization?
204. 🔴 Your GPU runs out of memory — list six ways to fit the model/training.
205. 🔴 How do you speed up data loading when the GPU is starved (bottlenecked on I/O)?
206. 🔴 Explain quantization-aware training vs post-training quantization.

**Ethics / robustness (increasingly asked)**
207. 🟡 How would you detect bias in a model's predictions across groups?
208. 🔴 What is an adversarial example and how do you defend against it?
209. 🔴 How do you handle PII and privacy in a training pipeline?

**Hard conceptual "why" questions (verbal)**
210. 🟡 Why does increasing model size sometimes reduce test error and sometimes overfit?
211. 🔴 Explain double descent.
212. 🔴 Why do we use cross-entropy instead of MSE for classification?
213. 🔴 Why can a model have low loss but poor calibration, and how do you fix calibration?
214. 🔴 Why does batch norm behave differently at train vs inference time?

---

That's **214 total** across four batches — genuinely comprehensive for an ML/DL engineer loop, from NumPy syntax up through FAANG system design and cutting-edge DL.

I'll be straight with you: you now have far more questions than time to practice them, and reading questions isn't the same as being able to *answer* them under pressure. The highest-value move before an interview is active recall — attempting answers and getting corrected.

So my strong recommendation: let's **drill**. I pick a question, you answer in your own words or code, and I grade + push on the "why." Even 10 drilled questions will help more than another 55 read.

Want to start drilling (I'll begin with a Topic 4 / PyTorch one), or is there a *specific* subtopic you'd rather I generate a batch 5 for?
