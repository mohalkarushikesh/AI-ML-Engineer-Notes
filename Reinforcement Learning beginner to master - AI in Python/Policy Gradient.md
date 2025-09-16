Policy Gradient methods are a class of reinforcement learning algorithms that **directly optimize the agent’s policy** — the strategy it uses to choose actions — rather than estimating value functions like Q-learning does.

---

### 🎯 Core Idea

Instead of learning how good each state or action is, policy gradients learn **how to act** by adjusting the parameters of a policy function to **maximize expected rewards**.

The objective is:
$$
J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_t R_t \right]
$$
Where:
- \( \pi_\theta \): the policy parameterized by \( \theta \)
- \( R_t \): reward at time step \( t \)

---

### 🔁 How It Works

1. **Rollout**: The agent interacts with the environment using its current policy.
2. **Collect Rewards**: It gathers states, actions, and rewards from the episode.
3. **Compute Gradient**: It calculates how changing the policy would affect the total reward.
4. **Update Policy**: It adjusts the policy parameters using **gradient ascent** to increase expected rewards.

---

### 🧠 Popular Variants

| Method                  | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| **REINFORCE**           | Monte Carlo-based; uses full episodes to update the policy. Simple but high variance. |
| **Actor-Critic**        | Combines a policy (actor) with a value function (critic) to reduce variance. |
| **PPO (Proximal Policy Optimization)** | Adds constraints to keep updates stable and efficient. Widely used in practice. |

---

### ✅ Advantages

- Handles **continuous action spaces** well.
- Learns **stochastic policies**, which are useful in uncertain environments.
- Can be combined with neural networks for **deep reinforcement learning**.

You can explore a detailed walkthrough on [GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/policy-gradient-methods-in-reinforcement-learning/) or check out a hands-on tutorial from [DataCamp](https://www.datacamp.com/tutorial/policy-gradient-theorem).
