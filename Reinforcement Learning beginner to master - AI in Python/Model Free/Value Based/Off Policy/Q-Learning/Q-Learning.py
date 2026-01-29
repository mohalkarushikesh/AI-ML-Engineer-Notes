# 1 Define the Enviroment
import numpy as np

n_states = 16
n_actions = 4
goal_state = 15

Q_table = np.zeros((n_states, n_actions))
# print(Q_table)

# 2 Set Hyperparameters
# Controls the how new information overrides the old. High value = fast learning
learning_rate = 0.8
# Determines how the futures rewards matter. close to 1 = long-term planning
discount_factor = 0.95
# 20% chances of exploring random actions. Balances the exploration vs. exploitation
exploration_prob = 0.2
# Number of training episodes. More epochs = more chances of refine the policy
epochs = 1000

action_effects = [-1, 1, -4, 4] # left, right, up, down in a 4x4 grid

# 3 Implement Q-Learning Algorithm
for epoch in range(epochs):
    # Start from the random state at the beginning of the each epoch
    current_state = np.random.randint(0, n_states)

    steps = 0

    # Continue until the agent reaches the goal state
    while current_state != goal_state:

        steps += 1

        # Decide weather to explore or exploit
        if np.random.rand() < exploration_prob:
            # Explore : Choose random action
            action = np.random.randint(0, n_actions)
        else:
            # Exploit : Choose best known action from the q-table
            action = np.argmax(Q_table[current_state])

        # Simulate the enviroment's response: move to the next state
        next_state = current_state + action_effects[action]
        next_state = max(0, min(next_state, n_states - 1)) # clamp to valid range

        # Assign the reward 1 if the goal is reached else 0
        reward = 1 if next_state == goal_state else 0

        # Q-Learning update rule
        # Update Q-Value for current state-action pair
        Q_table[current_state, action] += learning_rate * (reward + discount_factor * np.max(Q_table[next_state]) - Q_table[current_state, action])

        # Move to the next step
        current_state = next_state

    if epoch % 100 == 0 :
        print(f"Epoch {epoch}: reached goal in {steps} steps")

# 4 Output the Learned Q-Table
q_values_grid = np.max(Q_table, axis=1).reshape(4, 4)

print("Learned Q-table: ")
print(Q_table)

"""
Epoch 0: reached goal in 19627 steps
Epoch 100: reached goal in 6 steps
Epoch 200: reached goal in 5 steps
Epoch 300: reached goal in 2 steps
Epoch 400: reached goal in 3 steps
Epoch 500: reached goal in 1 steps
Epoch 600: reached goal in 7 steps
Epoch 700: reached goal in 6 steps
Epoch 800: reached goal in 2 steps
Epoch 900: reached goal in 4 steps
Learned Q-table:
[[0.8143039  0.81128263 0.81443106 0.857375  ]
 [0.80869918 0.8141908  0.81449124 0.857375  ]
 [0.81450625 0.857375   0.781926   0.6804128 ]
 [0.81416334 0.85737281 0.81287724 0.9025    ]
 [0.857375   0.85736875 0.81450625 0.9025    ]
 [0.85730641 0.85737493 0.81449186 0.9025    ]
 [0.85737478 0.90249769 0.77206005 0.9025    ]
 [0.85737499 0.9025     0.85737499 0.95      ]
 [0.9025     0.90249999 0.85737498 0.95      ]
 [0.90249942 0.90249998 0.85736917 0.95      ]
 [0.9024999  0.95       0.85737456 0.94967718]
 [0.9025     0.95       0.9025     1.        ]
 [0.95       0.95       0.90249769 1.        ]
 [0.94999998 0.94999997 0.9025     1.        ]
 [0.95       0.96       0.8664     1.        ]
 [0.         0.         0.         0.        ]]

"""
