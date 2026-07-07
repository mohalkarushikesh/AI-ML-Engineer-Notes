Here are hands-on mathematics practice exercises for AI/ML, organized by level. Each is meant to be worked through with code (NumPy is ideal) so you connect the math to what actually happens inside ML algorithms.

## Beginner

1. **Vector operations from scratch** — Implement dot product, magnitude, and cosine similarity with plain Python/NumPy; verify against library functions and interpret what cosine similarity means geometrically.
2. **Matrix multiplication by hand** — Multiply two matrices manually (nested loops), then confirm with NumPy; explain why the inner dimensions must match.
3. **Descriptive statistics** — Compute mean, median, variance, and standard deviation for a dataset by hand, then relate them to feature scaling.
4. **Probability basics** — Simulate coin flips and dice rolls to empirically verify probabilities; compute joint, marginal, and conditional probabilities from a small table.
5. **Bayes' theorem** — Build a simple spam-filter calculation: given P(word|spam) and priors, compute P(spam|word) step by step.
6. **Derivatives & slopes** — Numerically approximate the derivative of a function using finite differences and compare to the analytical derivative.
7. **Plotting functions** — Graph linear, quadratic, exponential, sigmoid, and ReLU functions; understand their shapes since they appear everywhere in ML.
8. **Gaussian distribution** — Sample from a normal distribution, plot the histogram, and overlay the theoretical density; vary mean and variance.

## Medium

1. **Gradient descent by hand** — Minimize a simple function like f(x) = x² manually, then extend to a 2D function; visualize the descent path on a contour plot.
2. **Linear regression via normal equation** — Derive and implement the closed-form solution (XᵀX)⁻¹Xᵀy; compare it to a gradient-descent solution.
3. **Eigenvalues & eigenvectors** — Compute them for small matrices by hand, verify with NumPy, and interpret them geometrically (directions that don't rotate).
4. **PCA from scratch** — Implement PCA using the covariance matrix and eigendecomposition; project data onto principal components and reconstruct it.
5. **Partial derivatives & gradients** — Compute the gradient of a multivariable function analytically, then verify numerically; connect this to loss surfaces.
6. **The chain rule** — Work through composite functions by hand, then show how it underlies backpropagation in a tiny 2-layer network.
7. **Cost functions** — Derive and plot MSE and cross-entropy loss; compute their gradients and explain why cross-entropy suits classification.
8. **Covariance & correlation matrices** — Build them from a multi-feature dataset; interpret what the off-diagonal values reveal about relationships.
9. **Maximum likelihood estimation** — Derive the MLE for the mean/variance of a Gaussian; verify by optimizing the log-likelihood numerically.
10. **Probability distributions in practice** — Fit Bernoulli, binomial, and Poisson distributions to data and compare observed vs. expected frequencies.

## Advanced

1. **Backpropagation from scratch** — Derive the gradients for every layer of a small neural network by hand and implement the full forward/backward pass in NumPy.
2. **Singular Value Decomposition (SVD)** — Implement or apply SVD; use it for low-rank matrix approximation and image compression, and relate it to PCA.
3. **Constrained optimization & Lagrange multipliers** — Solve a constrained problem analytically (e.g., the margin in SVMs) and connect it to the optimization ML solvers perform.
4. **Convex optimization** — Explore convex vs. non-convex functions; implement gradient descent with momentum and Adam and analyze convergence behavior.
5. **Information theory** — Compute entropy, cross-entropy, and KL divergence for distributions; use KL divergence to compare model predictions to targets.
6. **The Jacobian & Hessian** — Compute them for a vector-valued function; explain their roles in second-order optimization and in analyzing loss curvature.
7. **Matrix calculus for ML** — Derive gradients of matrix expressions (e.g., ∂(‖Xw − y‖²)/∂w) used throughout linear models and neural nets.
8. **Bayesian inference** — Implement a full Bayesian update with conjugate priors (e.g., Beta-Binomial); plot how the posterior evolves as data arrives.
9. **Markov chains & stationary distributions** — Build a transition matrix, simulate the chain, and compute the stationary distribution via eigen-analysis (the basis for MDPs in RL).
10. **Numerical stability** — Implement the log-sum-exp trick and a stable softmax; demonstrate how naive implementations overflow or lose precision.
11. **Optimization landscape visualization** — Plot loss surfaces for simple models, add saddle points and local minima, and compare how different optimizers navigate them.
12. **Central Limit Theorem simulation** — Empirically demonstrate the CLT by averaging samples from non-normal distributions and watching the sampling distribution approach normal.

A good approach is to always pair each concept with a small implementation and a visualization — the math sticks far better when you can *see* the gradient descending or the eigenvectors pointing along the data. Work one project per level fully before advancing.

Want me to expand any single topic into a full step-by-step exercise with the derivation, starter code, and a worked example?
