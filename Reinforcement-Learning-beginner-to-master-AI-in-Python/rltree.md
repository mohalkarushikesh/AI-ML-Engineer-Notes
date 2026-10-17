# Reinforcement Learning Taxonomy

## 1. Fundamentals
* **Markov Decision Process (MDP):** The mathematical framework (States, Actions, Rewards, Transitions).
* **POMDP:** Partially Observable MDPs (where the agent can't see the full state).

---

## 2. Model-Free RL (Learning by Trial and Error)
### A. Value-Based (Estimating the 'Value' of states/actions)
* **On-Policy:**
    * SARSA
* **Off-Policy:**
    * Q-Learning
    * DQN (Deep Q-Network)
    * Double DQN (Fixes overestimation bias)
    * Dueling DQN (Separates state value and action advantage)
    * **Distributional RL:** C51, QR-DQN, IQN (Predicts reward distributions, not just means)
    * **Rainbow DQN:** (Combines all the above improvements)

### B. Policy-Based (Directly optimizing the strategy)
* **On-Policy:**
    * REINFORCE (Monte Carlo Policy Gradient)
    * TRPO (Trust Region Policy Optimization)
    * PPO (Proximal Policy Optimization - *Industry Standard*)
* **Off-Policy:**
    * DDPG (Deep Deterministic Policy Gradient)
    * TD3 (Twin Delayed DDPG)
    * SAC (Soft Actor-Critic - *SOTA for continuous control*)

### C. Actor-Critic (The Hybrid approach)
* A2C / A3C (Advantage Actor-Critic)
* IMPALA (Importance Weighted Actor-Learner Architecture)

---

## 3. Model-Based RL (Learning a simulation/world model)
* **Background Planning:** Dyna-Q
* **Search & Lookahead:** Monte Carlo Tree Search (MCTS), AlphaZero
* **Latent World Models:** * World Models (Ha & Schmidhuber)
    * **Dreamer (V1, V2, V3):** (Learning entirely inside a "dream" or imagination)
* **Planning with Learned Models:** MuZero, MBPO (Model-Based Policy Optimization)

---

## 4. Offline RL (Learning from static datasets/logs)
* **Policy Constraints:** BCQ (Batch-Constrained Q-learning)
* **Regularization:** CQL (Conservative Q-Learning)
* **Sequence Modeling:** * **Decision Transformer:** (Treating RL as a language translation problem)
    * Trajectory Transformer

---

## 5. Modern Paradigms & Specialized Branches
* **RLHF (RL from Human Feedback):** Used to align LLMs (e.g., ChatGPT, Gemini).
* **Multi-Agent RL (MARL):** QMIX, MAPPO, MADDPG.
* **Hierarchical RL (HRL):** Option-Critic, HIRO (Learning sub-goals vs. high-level goals).
* **Meta-RL:** MAML, RL² (Learning to adapt to new tasks quickly).
* **Multi-Objective RL:** Optimizing for multiple conflicting rewards.

---

## 6. Exploration Strategies
* **Intrinsic Motivation:** Curiosity-driven (ICM), RND (Random Network Distillation).
* **Advanced Search:** Go-Explore (Solving "montezuma's revenge" style hard-exploration games).