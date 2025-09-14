**Deep Q-Learning (DQN)**
- It is method that uses deep learning to help machines help decision in complicated situations. 
- It's especially useful in environments where the number of possible situations called states is very large like in video games or robotics.

Key Challenges Addressed by Deep Q-Learning: 
1. High dimentional state spaces
2. Continuous input data
3. Scalability

Architecture of Deep Q-Networks 
A DQN consists of following components:
1. Neural Network
2. Experience Replay
3. Target Network
4. Loss Function:

$$
\mathcal{L}(\theta) = \mathbb{E} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]
$$

---

### 🧠 What Each Term Means

| Symbol | Meaning |
|--------|---------|
| $\theta$ | Parameters of the current Q-network |
| $\theta^-$ | Parameters of the **target** Q-network (fixed for stability) |
| $Q(s, a; \theta)$ | Predicted Q-value for current state-action pair |
| $r$ | Reward received after taking action $a$ in state $s$ |
| $\gamma$ | Discount factor for future rewards |
| $\max_{a'} Q(s', a'; \theta^-)$ | Maximum predicted Q-value for next state $s'$, using target network |
| $\mathbb{E}$ | Expectation over the experience replay buffer |

---

### 🔍 What It Does

- This loss measures the **squared difference** between:
  - The **target Q-value**: $r + \gamma \max Q(s', a'; \theta^-)$
  - The **predicted Q-value**: $Q(s, a; \theta)$

- The goal is to **minimize this loss** by adjusting $\theta$, so the network better predicts future rewards.

Traning process of Deep Q-Learning:
1. Intialization
2. Exploration vs. Exploitation
3. Experience Collection
4. Training Updates
5. Target network update
6. Decay Exploration Rate

Applications of DQN
1. Atari games
2. Robotics
3. Self driving cars
4. finance
5. helthcare 
