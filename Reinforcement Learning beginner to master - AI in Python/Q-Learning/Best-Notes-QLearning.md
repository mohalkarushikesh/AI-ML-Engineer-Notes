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
