# Machine Learning (Classical) — In-Depth, with Pseudocode

> Scope: the core *classical* ML algorithms — the complement to the deep-learning reference. Each entry gives the math, language-neutral pseudocode, and the "what / why / when". NumPy-ish notation: `@` = matrix multiply, `·` = dot product, `*` = elementwise. `m` = #samples, `p` = #features, `X` = [m×p], `y` = targets.

**Map of the field**

```
Supervised ── Regression:  Linear, Ridge/Lasso, SVR, trees, boosting
           └─ Classification: Logistic, kNN, Naive Bayes, SVM, trees, RF, boosting
Unsupervised ── Clustering: k-Means, Hierarchical, DBSCAN, GMM
             └─ Dim. reduction: PCA, LDA, t-SNE/UMAP
Ensembles ── Bagging (RF), Boosting (AdaBoost/GBM/XGBoost), Stacking, Voting
Other ── Recommenders (matrix factorization), Association rules (Apriori)
```

---

## 0. The common skeleton

Most ML = **define a model with parameters → define a loss → minimize it** (closed-form or iterative), then regularize and validate.

```
model   = hypothesis(θ)                 # e.g. linear, tree, kernel
loss    = L(y, model(X)) + λ * penalty(θ)   # fit + regularization
θ*      = argmin_θ loss                   # normal equation, GD, EM, greedy splits...
predict = model_θ*(x_new)
```

---

## 1. Gradient descent (the workhorse optimizer)

```
def gradient_descent(X, y, lr, iters):
    θ = zeros(p)
    for _ in range(iters):
        g = (1/m) * gradient_of_loss(θ, X, y)   # average gradient
        θ = θ - lr * g
    return θ
# Batch  = all m samples per step (stable, slow)
# SGD    = 1 sample per step (noisy, fast, escapes local minima)
# Mini-batch = b samples (the practical default)
```

Convex losses (linear/logistic/SVM) → one global minimum. Scale features first so GD converges evenly.

---

## 2. Linear regression

**What:** fit `ŷ = Xθ`. **When:** continuous target, roughly linear relationship, interpretable baseline.

```
# Closed form (normal equation) — exact, O(p³), fine for small p
θ = inv(X.T @ X) @ X.T @ y

# Gradient descent — scales to large m, p
# loss = (1/2m) Σ (xᵢ·θ − yᵢ)²
g = (1/m) * X.T @ (X @ θ − y)
θ = θ − lr * g
```

**Assumptions:** linearity, independent errors, homoscedasticity, low multicollinearity. Check residual plots.

---

## 3. Regularized linear models

Add a penalty to shrink weights → reduce variance/overfitting.

| Model | Penalty | Effect |
|---|---|---|
| **Ridge (L2)** | `λ‖θ‖²` | Shrinks all weights smoothly; handles multicollinearity |
| **Lasso (L1)** | `λ‖θ‖₁` | Drives some weights to **0** → feature selection |
| **Elastic Net** | `λ₁‖θ‖₁ + λ₂‖θ‖²` | Mix of both |

```
# Ridge closed form
θ = inv(X.T @ X + λ*I) @ X.T @ y
# Ridge gradient step: g = (1/m) X.T (Xθ − y) + 2λθ
# Lasso: no closed form → coordinate descent or subgradient (soft-thresholding)
```

Bigger `λ` → more shrinkage → more bias, less variance. Tune `λ` by cross-validation.

---

## 4. Logistic regression (classification)

**What:** linear model + sigmoid → class probability. **When:** binary (or multiclass via softmax) classification, interpretable, strong baseline.

```
def predict_proba(X, θ):
    return sigmoid(X @ θ)              # sigmoid(z) = 1/(1+e^-z)

# Loss = binary cross-entropy:  −(1/m) Σ [ yᵢ log hᵢ + (1−yᵢ) log(1−hᵢ) ]
# Gradient has the SAME clean form as linear regression:
h = sigmoid(X @ θ)
g = (1/m) * X.T @ (h − y)
θ = θ − lr * g
# Multiclass: softmax regression (one weight vector per class).
# Decision: class 1 if h ≥ threshold (default 0.5; tune for precision/recall).
```

---

## 5. k-Nearest Neighbors (kNN)

**What:** no training — predict from the k closest stored points. **When:** small/medium data, low dimensions, nonlinear boundaries. **Cost:** slow at inference, suffers the curse of dimensionality.

```
def knn_predict(x, X_train, y_train, k):
    dists = [distance(x, xi) for xi in X_train]   # Euclidean / Manhattan / cosine
    idx   = argsort(dists)[:k]                     # k nearest
    return majority_vote(y_train[idx])             # or mean(y_train[idx]) for regression
```

Scale features (distance-based!). Use KD-tree/Ball-tree or ANN indexes to speed up search.

---

## 6. Naive Bayes

**What:** Bayes' rule + the "naive" assumption that features are conditionally independent. **When:** text/spam classification, high-dimensional sparse data, tiny training sets; very fast.

```
# P(y|x) ∝ P(y) * Π_i P(xᵢ | y)
def train(X, y):
    priors = {c: count(y==c)/m for c in classes}
    # Gaussian NB: store mean,var per (feature,class)
    # Multinomial NB (text): store word-count likelihoods per class (+ Laplace smoothing)

def predict(x):
    return argmax_c [ log P(c) + Σ_i log P(xᵢ | c) ]   # log-space avoids underflow
```

---

## 7. Decision trees (CART)

**What:** recursively split feature space to maximize purity. **When:** need interpretability, nonlinear, mixed feature types, no scaling needed. **Weakness:** high variance (overfits) alone.

```
def build_tree(data, depth):
    if stopping_criterion(data, depth):           # max_depth, min_samples, pure node
        return Leaf(prediction = majority_or_mean(data))
    best = argmax over (feature f, threshold t):
               impurity_decrease(split data by f ≤ t)
    left, right = split(data, best.f, best.t)
    return Node(best.f, best.t, build_tree(left, depth+1), build_tree(right, depth+1))

# Impurity (classification):
#   Gini    = 1 − Σ_k p_k²
#   Entropy = − Σ_k p_k log p_k
# Split gain = impurity(parent) − Σ (n_child/n) * impurity(child)
# Regression split: minimize child variance / MSE.
```

Control overfitting with max_depth, min_samples_leaf, or **post-pruning** (cost-complexity).

---

## 8. Ensembles — bagging, boosting, stacking

**Core idea:** combine many weak/varied models → lower error than any single one.

| Method | How | Reduces | Example |
|---|---|---|---|
| **Bagging** | Train models on bootstrap samples **in parallel**, average | **Variance** | Random Forest |
| **Boosting** | Train models **sequentially**, each fixing prior errors | **Bias** | AdaBoost, GBM, XGBoost |
| **Stacking** | Train a meta-model on base models' predictions | Both | Blend diverse models |
| **Voting** | Hard (majority) / soft (avg prob) across models | Variance | Quick ensemble |

### 8a. Random Forest (bagging of trees)
```
def random_forest_train(X, y, B):
    trees = []
    for b in 1..B:
        X_b, y_b = bootstrap_sample(X, y)              # sample with replacement
        tree = build_tree(X_b, y_b,
                          features_per_split = sqrt(p)) # random feature subset → decorrelate
        trees.append(tree)
    return trees

predict = majority_vote([t(x) for t in trees])          # or average for regression
# Bonus: out-of-bag (OOB) samples give a free validation estimate.
```

### 8b. AdaBoost
```
w = [1/m]*m                                     # sample weights
for t in 1..T:
    h_t  = fit_weak_learner(X, y, sample_weights=w)
    err  = Σ w_i * [h_t(x_i) ≠ y_i] / Σ w_i
    α_t  = 0.5 * log((1−err)/err)               # learner weight
    w_i *= exp(−α_t * y_i * h_t(x_i)); normalize # up-weight misclassified
final(x) = sign( Σ α_t * h_t(x) )
```

### 8c. Gradient Boosting (GBM / XGBoost / LightGBM)
```
F_0(x) = argmin_c Σ L(y_i, c)                   # e.g. mean(y) for MSE
for m in 1..M:
    r_i = − ∂L(y_i, F(x_i)) / ∂F(x_i)           # pseudo-residuals (= y−F for MSE)
    h_m = fit_regression_tree(X, r_i)           # fit a tree to the residuals
    F   = F + ν * h_m                            # ν = learning rate (shrinkage, e.g. 0.1)
return F
# XGBoost/LightGBM add: 2nd-order (Hessian) splits, L1/L2 reg, column/row subsampling,
# histogram binning, and clever handling of missing values → top tabular performance.
```

**Boosting intuition:** each new model learns what's still wrong (the gradient of the loss), so the ensemble steadily reduces bias.

---

## 9. Support Vector Machines (SVM)

**What:** find the **maximum-margin** separating boundary; the **kernel trick** handles nonlinearity. **When:** medium data, high dimensions (text), clear margins.

```
# Primal (soft margin): minimize  ½‖w‖² + C Σ ξ_i
#   subject to  y_i (w·x_i + b) ≥ 1 − ξ_i,  ξ_i ≥ 0
# Equivalent unconstrained (hinge loss) for SGD:
loss = ½‖w‖² + C Σ max(0, 1 − y_i (w·x_i + b))

# Kernel trick: replace dot products x·x' with K(x,x') to work in higher-dim space
#   Linear:      K = x·x'
#   Polynomial:  K = (x·x' + c)^d
#   RBF/Gaussian:K = exp(−γ‖x − x'‖²)     ← common default
```

`C` trades margin width vs. misclassification (small C = wider margin, more tolerance). Only **support vectors** (points on/inside the margin) define the boundary. Scale features.

---

## 10. Clustering (unsupervised)

### 10a. k-Means (Lloyd's algorithm)
**When:** spherical, similar-size clusters, you know k. Fast, scalable.
```
init k centroids                                 # k-means++ for good seeds
repeat:
    # Assignment step
    for each point x: label(x) = argmin_c ‖x − μ_c‖²
    # Update step
    for each cluster c: μ_c = mean(points with label c)
until labels stop changing (or max iters)
# Objective: minimize within-cluster sum of squares (inertia).
# Choose k: elbow method, silhouette score. Scale features.
```

### 10b. Hierarchical (agglomerative)
**When:** want a dendrogram / nested clusters, don't know k upfront.
```
start: each point is its own cluster
repeat:
    merge the two closest clusters           # linkage: single/complete/average/ward
until one cluster remains
# Cut the dendrogram at a height → choose number of clusters.
```

### 10c. DBSCAN (density-based)
**When:** arbitrary shapes, noise/outliers, unknown k.
```
for each unvisited point p:
    mark p visited
    N = region_query(p, eps)                 # points within eps
    if |N| < minPts: label p = NOISE
    else:
        start new cluster; add p
        expand: for each q in N, if q has ≥ minPts neighbors, add its neighbors too
# Two params: eps (radius), minPts (density). Finds non-convex clusters; labels outliers.
```

### 10d. Gaussian Mixture Models (EM)
**When:** soft (probabilistic) assignments, elliptical clusters.
```
init means μ_k, covariances Σ_k, weights π_k
repeat:
    # E-step: responsibility of cluster k for point i
    γ_ik = π_k N(x_i | μ_k, Σ_k) / Σ_j π_j N(x_i | μ_j, Σ_j)
    # M-step: re-estimate params weighted by responsibilities
    N_k = Σ_i γ_ik
    μ_k = (1/N_k) Σ_i γ_ik x_i
    Σ_k = (1/N_k) Σ_i γ_ik (x_i−μ_k)(x_i−μ_k)ᵀ
    π_k = N_k / m
until log-likelihood converges
```
EM is the general recipe: alternate "guess the hidden assignments" (E) and "optimize params given them" (M).

---

## 11. Dimensionality reduction

### 11a. PCA (linear, unsupervised)
**What:** project onto directions of maximum variance. **When:** compress features, de-correlate, visualize, denoise.
```
X = X − mean(X)                     # center
C = (1/m) * X.T @ X                  # covariance matrix (p×p)
eigvals, eigvecs = eig(C)           # or use SVD: X = U S Vᵀ, components = V
order by eigval desc; take top k eigvecs → W  (p×k)
Z = X @ W                            # projected data (m×k)
# Keep enough components for ~95% explained variance (Σ top eigvals / Σ all).
```

### 11b. Others
- **LDA** — *supervised* projection maximizing class separation (uses labels).
- **t-SNE / UMAP** — nonlinear, for **visualization** of high-dim data in 2–3D (preserve local structure; don't use distances/axes quantitatively; UMAP is faster and preserves more global structure).

---

## 12. Recommenders — matrix factorization

**What:** factor the user–item rating matrix `R ≈ P Qᵀ` into latent factors. **When:** collaborative filtering at scale.
```
# minimize Σ_(u,i observed) (r_ui − p_u·q_i)² + λ(‖p_u‖² + ‖q_i‖²)
for each observed rating (u, i, r_ui):
    e = r_ui − p_u · q_i
    p_u += lr * (e * q_i − λ * p_u)
    q_i += lr * (e * p_u − λ * q_i)
predict r_ui = p_u · q_i
# Alternatives: user/item-based kNN (memory-based CF), ALS (alternating least squares).
```

(Apriori / FP-Growth mine **association rules** — "customers who bought X also bought Y" — via support/confidence/lift thresholds.)

---

## 13. Core ML machinery (applies to all)

### Bias–variance tradeoff
- **High bias (underfit):** too simple → bad on train & test. Fix: richer model, more features, less regularization.
- **High variance (overfit):** memorizes train → bad on test. Fix: more data, regularization, simpler model, ensembling, cross-validation.
- Total error ≈ bias² + variance + irreducible noise.

### Validation
```
train / validation / test split          # test touched ONCE, at the end
k-fold cross-validation:                  # robust estimate for small data
    split into k folds; train on k−1, validate on 1; rotate; average
# Stratified k-fold for imbalanced classes. Time-series: use forward-chaining splits.
```

### Metrics
- **Classification:** accuracy, precision, recall, F1, ROC-AUC, PR-AUC, log-loss, confusion matrix. FP-costly → precision; FN-costly → recall; imbalanced → F1/AUC.
- **Regression:** MAE, MSE/RMSE, R², MAPE.
- **Clustering:** silhouette, Davies–Bouldin, (ARI/NMI if labels known).

### Feature engineering & preprocessing
- **Scaling:** standardize (z-score) or normalize for distance/gradient methods (kNN, SVM, k-Means, linear); trees don't need it.
- **Encoding:** one-hot (low cardinality), ordinal, target/mean encoding (high cardinality).
- **Missing values:** drop, impute (mean/median/mode/model), or let tree methods handle them.
- **Imbalance:** SMOTE/oversample, undersample, class weights, threshold tuning.
- **Beware leakage:** fit scalers/encoders on **train only**; no target-derived features from the future.

### Hyperparameter tuning
- **Grid search** (exhaustive), **Random search** (efficient in high-dim), **Bayesian optimization** (smart, e.g. Optuna/Hyperopt). Always tune against validation folds, never the test set.

---

## 14. Choosing an algorithm

| Situation | Reach for |
|---|---|
| Tabular, want top accuracy | **Gradient boosting** (XGBoost/LightGBM/CatBoost) |
| Need interpretability | Linear/Logistic regression, single decision tree |
| Small data, nonlinear | kNN, SVM (RBF) |
| High-dim sparse / text | Linear SVM, Multinomial Naive Bayes, Logistic regression |
| Robust, low-tuning baseline | **Random Forest** |
| Feature selection built in | Lasso, tree importances |
| Cluster unknown-k, odd shapes | DBSCAN |
| Cluster known-k, fast | k-Means |
| Soft/probabilistic clusters | GMM |
| Compress / visualize features | PCA (linear), UMAP/t-SNE (viz) |
| Recommendations | Matrix factorization / ALS |
| Huge data, online learning | SGD-based linear/logistic |

**General order of attack:** strong simple baseline (logistic/linear or RF) → gradient boosting for tabular → tune + validate → only reach for deep learning if you have lots of data or unstructured inputs (images/text/audio).

---

### Highest-yield takeaways
1. **Most ML = model + loss + optimizer (+ regularizer)**; linear/logistic/SVM are convex (one global optimum).
2. **Logistic regression's gradient = linear regression's gradient** — `(1/m) Xᵀ(h − y)`; only the hypothesis changes.
3. **Regularization:** L2 (Ridge) shrinks smoothly; L1 (Lasso) zeros features (selection); tune λ by CV.
4. **Trees** are interpretable but high-variance → **bagging (Random Forest)** cuts variance, **boosting (GBM/XGBoost)** cuts bias.
5. **Gradient boosting fits each new tree to the residual gradients** — usually the best off-the-shelf tabular model.
6. **SVM** maximizes margin; the **kernel trick (RBF)** gives nonlinearity; only support vectors matter.
7. **Clustering picks:** k-Means (spherical, known k), DBSCAN (shapes + noise), GMM (soft), hierarchical (dendrogram).
8. **EM = alternate E-step (estimate hidden assignments) and M-step (optimize params)** — powers GMM and more.
9. **PCA** = project onto max-variance directions (eigenvectors of the covariance / SVD).
10. **Scale distance/gradient methods, never leak the test set, and judge imbalanced problems by F1/AUC, not accuracy.**
