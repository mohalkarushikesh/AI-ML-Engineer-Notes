import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Create FrozenLake environment (slippery, no rendering for speed)
env = gym.make("FrozenLake-v1", is_slippery=True)

# Initialize Q-table
qtable = np.zeros((env.observation_space.n, env.action_space.n))

# Hyperparameters
episodes = 5000
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.001
min_epsilon = 0.01
max_steps = 100

# Track outcomes
outcomes = []

print("Q-table before training:\n", qtable)

# Training loop
for episode in range(episodes):
    state, _ = env.reset()
    done = False
    steps = 0
    outcomes.append('Failure')

    while not done and steps < max_steps:
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(qtable[state])

        # Take action
        new_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Optional reward shaping
        if reward == 0 and not done:
            reward = -0.01

        # Q-learning update
        qtable[state, action] += alpha * (reward + gamma * np.max(qtable[new_state]) - qtable[state, action])
        state = new_state
        steps += 1

        if reward > 0:
            outcomes[-1] = 'Success'

    # Adaptive epsilon decay
    if outcomes[-1] == 'Success':
        epsilon *= 0.99
    else:
        epsilon *= 1.01
    epsilon = np.clip(epsilon, min_epsilon, 1.0)

print("\n======================================")
print("Q-table after training:\n", qtable)

# Plot training outcomes
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 17})
plt.figure(figsize=(12, 5))
plt.xlabel('Episode')
plt.ylabel('Outcome')
ax = plt.gca()
ax.set_facecolor('#efeeea')
colors = ['#FF4C4C' if outcome == 'Failure' else '#0A047A' for outcome in outcomes]
plt.bar(range(len(outcomes)), [1]*len(outcomes), color=colors, width=1.0)
plt.yticks([])
plt.title("Training outcomes: Success vs Failure")
plt.show()

# Evaluation
eval_episodes = 500
nb_success = 0

for _ in range(eval_episodes):
    state, _ = env.reset()
    done = False

    while not done:
        action = np.argmax(qtable[state])
        new_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = new_state
        nb_success += reward

print(f"Success rate after training = {nb_success / eval_episodes * 100:.2f}%")

env.close()
