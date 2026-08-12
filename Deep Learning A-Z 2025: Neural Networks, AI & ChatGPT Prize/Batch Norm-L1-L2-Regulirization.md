**Batch normalization (simple):**
Imagine each layer is a student getting notes from the previous layer. If those notes keep changing in scale and range every batch, the student struggles to learn. Batch norm cleans up the notes — rescales them to a consistent mean 0, variance 1 — so every layer gets stable, predictable input. Then γ and β let the network adjust that if it wants. Result: trains faster, less fussy about initialization.

*One-liner: "Keeps the numbers flowing between layers on a consistent scale so training is smoother and faster."*

---

**L1 and L2 regularization (simple):**
Both fight overfitting by adding a penalty on large weights to the loss — punishing the model for being too complex.

- **L2 (Ridge)** — penalizes the **square** of weights. Pushes all weights *small but not zero*. Spreads influence across features. Think: "shrink everyone a little."
- **L1 (Lasso)** — penalizes the **absolute value** of weights. Pushes many weights *exactly to zero* → automatic feature selection. Think: "drop the useless features entirely."

Quick memory hook:
- **L1 → sparse** (zeros out features, picks the important ones)
- **L2 → small** (shrinks everything smoothly, keeps all features)

*One-liner: "Both penalize big weights to prevent overfitting — L1 zeroes out weak features (selection), L2 just shrinks all weights smoothly."*

Why the difference? L1's absolute-value penalty has a sharp corner at zero that "snaps" weights to exactly 0; L2's smooth curve only ever shrinks them toward 0 but never quite there.
