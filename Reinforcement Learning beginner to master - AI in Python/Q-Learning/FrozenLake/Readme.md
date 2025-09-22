## FrozenLake Navigation (Q-Learning)

This README explains the FrozenLake environment and a simple Q-Learning approach to navigate it. The goal is to teach an agent to reach the goal tile while avoiding holes.

### Environment Overview

- The FrozenLake environment is a grid of tiles.
- Tiles can be safe frozen surfaces or dangerous holes.
- The agent can take 4 actions: left, right, up, down.
- The episode ends when the agent reaches the goal or falls into a hole.

Default 4x4 layout:

```
S F F F   (S: starting point, safe)
F H F H   (F: frozen surface, safe)
F F F H   (H: hole, terminal)
H F F G   (G: goal, terminal)
```

### Objective

Reach the goal in as few steps as possible without falling into holes. The agent learns a policy by estimating action values in a Q-table.

### Q-Learning Recap

- Initialize a Q-table with zeros: shape = [num_states, num_actions].
- Repeat for episodes:
  - Start at an initial state.
  - Choose an action using an exploration policy (e.g., ε-greedy).
  - Observe reward and next state.
  - Update Q-value using: 
    Q(s, a) ← Q(s, a) + α [ r + γ max_a' Q(s', a') − Q(s, a) ]
  - Move to next state; stop on terminal.

Key hyperparameters:
- α (learning rate): how strongly new information overrides old.
- γ (discount factor): weight of future rewards.
- ε (epsilon): exploration rate for ε-greedy policy (often decayed over time).

### Example Q-Table Output

Q-table before training:
```
[[0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]]
```

Q-table after training:
```
[[-2.62596647e-02 -2.16368924e-02 -2.08548299e-02 -2.70266122e-02]
 [-1.74586318e-02 -1.09145124e-02 -7.16232815e-03 -1.84335276e-02]
 [-4.54144442e-03 -9.88400553e-03  7.03606563e-03 -1.51779977e-02]
 [-1.22194391e-02 -1.09400811e-02 -1.68930813e-02 -1.75668186e-02]
 [-2.12527331e-02 -8.07855590e-03 -8.73487470e-03 -1.45466019e-02]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00]
 [ 2.46581686e-02  1.93031624e-02  1.52614667e-02 -4.56533501e-03]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00]
 [-9.98802879e-03  1.58240567e-02  6.12827503e-03  2.16756867e-03]
 [ 2.05810690e-02  6.16362650e-02  3.66358635e-02 -1.10805880e-03]
 [ 6.48038579e-02  1.09670813e-01  6.84057021e-02  2.92216770e-04]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00]
 [ 5.78142576e-03  9.21001978e-02  1.20549809e-01  4.27338277e-02]
 [ 3.14042983e-02  3.88103397e-01  3.72993762e-01  1.69791828e-01]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00]]
```

Success rate after training = 6.00%

Note: FrozenLake with default slippery dynamics can be challenging for tabular Q-Learning; success rates may vary widely with hyperparameters, exploration schedules, and number of episodes. Consider using more episodes, ε decay, or the non-slippery variant for clearer convergence.

### How to Reproduce

- Dependencies: `gymnasium` (or `gym` for older code), `numpy`.
- Typical steps:
  1. Create environment: `gymnasium.make("FrozenLake-v1", is_slippery=True)`.
  2. Initialize Q-table with zeros `[env.observation_space.n, env.action_space.n]`.
  3. Run training loop for N episodes with ε-greedy policy.
  4. Evaluate the learned policy over test episodes to compute success rate.

Example hyperparameters to try:
- α = 0.8 → 0.1 (decay over time)
- γ = 0.95
- ε = 1.0 → 0.01 (exponential decay)
- Episodes: 5,000–50,000 (more for slippery env)

### Tips

- For faster learning, try `is_slippery=False`.
- Use ε decay to balance exploration and exploitation.
- Aggregate rewards and periodically print average success to monitor progress.
- Seed the environment for reproducibility when comparing settings.
