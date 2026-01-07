**Reinforcement Learning (RL) cheatsheet** 🧠📘 

## 🧠 Core Concepts
- **Agent** → Learner/decision-maker.
- **Environment** → Where the agent operates.
- **State (s)** → Snapshot of the environment.
- **Action (a)** → Decision taken by agent.
- **Reward (r)** → Feedback signal.
- **Policy (π)** → Strategy mapping states to actions.
- **Value Function (V)** → Expected reward from a state.
- **Q-Function (Q)** → Expected reward from state-action pair.

---

## 🔁 Learning Types
- **Model-Free** → Learns from experience (e.g., Q-learning).
- **Model-Based** → Builds model of environment.
- **On-Policy** → Learns from its own moves/current policy (e.g., SARSA).
- **Off-Policy** → Learns from exploring + greedy (e.g., Q-learning).

---

## 🧩 Key Algorithms
- **Dynamic Programming** → Value Iteration, Policy Iteration.
- **Monte Carlo Methods** → Learn from episodes.
- **TD Learning**:  
  ‣ TD(0) — simplest update  
  ‣ SARSA — on-policy  
  ‣ Q-learning — off-policy 
- **Policy Gradient**:  
  ‣ REINFORCE — pure gradient  
  ‣ Actor-Critic — combines value + policy  
- **Deep RL**:  
  ‣ DQN — deep Q-learning  
  ‣ DDPG — continuous actions  
  ‣ PPO — stable policy updates  
  ‣ A3C — async agents  
  ‣ SAC — entropy-regularized  

---

## 🛠️ Libraries & Tools
- **OpenAI Gym** → RL environments.
- **Stable Baselines3** → Prebuilt RL algorithms.
- **RLlib** → Scalable RL.
- **PettingZoo** → Multi-agent RL.
- **TensorFlow / PyTorch** → Deep learning backend.

---

## 📐 Evaluation
- **Cumulative Reward** → Total reward over time.
- **Average Reward** → Mean per episode.
- **Convergence** → Stability of learning.
- **Exploration vs Exploitation** → Balance discovery and reward.

---

## 🚀 RL Workflow
1. Define environment  
2. Choose agent and algorithm  
3. Train agent via episodes  
4. Evaluate performance  
5. Tune hyperparameters  
6. Deploy or simulate  

---
