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

======================================

Q-table after training:
```
[[0.17894983 0.18128311 0.16399965 0.15059197]
 [0.07229203 0.07817644 0.0868104  0.12404711]
 [0.12473482 0.10474777 0.10379051 0.08239553]
 [0.04239828 0.05364193 0.04015067 0.05724118]
 [0.22861185 0.15442403 0.14914827 0.1268581 ]
 [0.         0.         0.         0.        ]
 [0.1924347  0.10859681 0.19897745 0.02827021]
 [0.         0.         0.         0.        ]
 [0.14345881 0.25527413 0.17108618 0.32410361]
 [0.24036895 0.43524972 0.34508427 0.29232213]
 [0.43268862 0.36427768 0.42792173 0.17164618]
 [0.         0.         0.         0.        ]
 [0.         0.         0.         0.        ]
 [0.15656475 0.40557768 0.5490125  0.49973795]
 [0.57630448 0.71103242 0.74655957 0.64946016]
 [0.         0.         0.         0.        ]]
```

Success rate after training = 43.80%

### Training result

<img width="1920" height="967" alt="Training_Outcomes" src="https://github.com/user-attachments/assets/dbc53fa0-6061-411d-a130-31b8e16f0485" />

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
