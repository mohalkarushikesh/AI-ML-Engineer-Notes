The **Actor-Critic method** is a powerful approach in reinforcement learning that combines the strengths of two key ideas: **policy-based** and **value-based** learning.

---

### 🎭 Two Main Roles

1. **Actor**  
   - Learns the **policy**: decides which action to take in a given state.  
   - Tries to **maximize rewards** by improving its decision-making over time.  
   - Outputs probabilities for actions (π(a|s)).

2. **Critic**  
   - Learns the **value function**: evaluates how good the current state or action is.  
   - Provides feedback to the actor by estimating the **expected future rewards**.  
   - Computes the **advantage** or **TD error** to guide the actor’s updates.

---

### 🔁 How It Works (Simplified Flow)

1. The **actor** picks an action based on the current policy.
2. The environment returns a reward and a new state.
3. The **critic** evaluates the action by estimating how good it was.
4. The actor uses this feedback to adjust its policy.
5. The critic updates its value estimates based on the observed reward.

---

### 🧠 Why Use Actor-Critic?

- Combines **exploration** (actor) and **evaluation** (critic).
- More **sample-efficient** than pure policy gradient methods.
- Works well with **continuous action spaces**.
- Can be implemented with neural networks for deep RL.

---
