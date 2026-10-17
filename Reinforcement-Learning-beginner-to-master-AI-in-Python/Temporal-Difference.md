Temporal Difference (TD) learning is a foundational concept in reinforcement learning that blends ideas from **Monte Carlo methods** and **Dynamic Programming**. It allows agents to learn directly from experience — without needing a model of the environment — and updates estimates **after every step**, not just at the end of an episode.

---

### 🧠 What Is Temporal Difference Learning?

TD learning estimates the value of a state by combining:
- The **actual reward** received after taking an action.
- The **predicted value** of the next state.

This is known as **bootstrapping**, because it updates predictions based on other predictions.

---

### 🔁 TD(0) Update Rule

For a state-value function $V(s)$ , the update rule is:

$$
V(s_t) \leftarrow V(s_t) + \alpha \left[ r_{t+1} + \gamma V(s_{t+1}) - V(s_t) \right]
$$

Where:
- $\alpha$ : learning rate
- $\gamma$ : discount factor
- $r_{t+1}$ : reward received after taking action
- $V(s_{t+1})$ : estimated value of the next state

The term inside the brackets is called the **TD error**:

$$
\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)
$$

---

### ✅ Why TD Learning Is Powerful

- **Model-free**: No need to know transition probabilities.
- **Online updates**: Learns after every step, not just after full episodes.
- **Efficient**: Works well for ongoing tasks like games or robotics.

---

You can explore more in-depth explanations on [GeeksforGeeks](https://www.geeksforgeeks.org/deep-learning/temporal-difference-td-learning/) or [TutorialsPoint](https://www.tutorialspoint.com/machine_learning/machine_learning_temporal_difference_learning.htm).

