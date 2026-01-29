[Q-Learning-Resource](https://medium.com/@alwinraju/in-depth-guide-to-implementing-q-learning-in-python-with-openai-gyms-taxi-environment-cd356cc6a288)

The Q-learning update rule is:


Where:

• Q(s, a): The current estimate of the Q-value for state s and action a.

• Alpha (α): The learning rate, determining how much new information overrides old information.

• R: The reward received after taking action a in state s.

• Gamma (γ): The discount factor, indicating how much future rewards are valued over immediate rewards.

• s’ (s prime): The new state after taking action a.

• Max over all a’ of Q(s’, a’): The maximum expected future reward for state s’.

- Learning Rate (alpha): Determines to what extent newly acquired information overrides old information.

- Discount Factor (gamma): Measures the importance of future rewards.

- Exploration Rate (epsilon): Controls the trade-off between exploration (trying new actions) and exploitation (using known actions).

- Temporal Difference (TD) Target: The expected reward of the current action plus the discounted future rewards.

- TD Error: The difference between the TD target and the current Q-value.

```
def update_q_table(state, action, reward, next_state):
    best_next_action = np.argmax(q_table[next_state, :])
    td_target = reward + gamma * q_table[next_state, best_next_action]
    td_error = td_target - q_table[state, action]
    q_table[state, action] += alpha * td_error
```

- Episode Loop: We run multiple episodes to allow the agent to learn from different starting positions.

- Step Loop: For each step in an episode, the agent chooses an action, observes the outcome, and updates the Q-table.

- Epsilon Decay: Gradually reduce exploration over time.

- Epsilon-Greedy Strategy Explained

• Exploration: Allows the agent to discover new states by taking random actions.
• Exploitation: Utilizes the knowledge accumulated in the Q-table to make the best decision.
• Balance: Starting with a high epsilon encourages exploration; decaying epsilon over time shifts focus to exploitation.

The Q-Learning Update Rule

The update rule adjusts the Q-value towards the TD target:

• Learning Rate ( \alpha ): Controls how much the new information affects the existing Q-value.
• Discount Factor ( \gamma ): Emphasizes the importance of future rewards.

Decaying Epsilon

epsilon = max(epsilon_min, epsilon * epsilon_decay)
• Ensures that epsilon does not fall below epsilon_min.

• Gradually reduces the exploration rate to focus on exploitation as the agent learns.
