import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Hyperparameters
episodes = 10000
alpha = 0.9
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.9995
epsilon_min = 0.01
max_steps = 100

# Initialize environment
env = gym.make("Taxi-v3")
action_size = env.action_space.n
state_size = env.observation_space.n
q_table = np.zeros((state_size, action_size))

# Track training rewards
training_rewards = []

# Epsilon-greedy action selection
def choose_action(state, epsilon):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state])

# Q-table update
def update_q_table(state, action, reward, next_state):
    best_next_action = np.argmax(q_table[next_state])
    td_target = reward + gamma * q_table[next_state, best_next_action]
    td_error = td_target - q_table[state, action]
    q_table[state, action] += alpha * td_error

# Moving average for plotting
def moving_average(data, window_size=100):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Training loop
for episode in range(episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0

    for step in range(max_steps):
        action = choose_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        update_q_table(state, action, reward, next_state)
        state = next_state
        total_reward += reward

        if done:
            break

    training_rewards.append(total_reward)
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    alpha = max(0.1, alpha * 0.999)

    if episode % 500 == 0:
        avg_reward = np.mean(training_rewards[-100:])
        print(f"Episode {episode}, Epsilon: {epsilon:.3f}, Avg Reward (last 100): {avg_reward:.2f}")

    if episode > 500 and np.mean(training_rewards[-100:]) > 8:
        print("Early stopping: agent consistently performs well.")
        break

print("\nTraining complete!")
print("Final Q-table snapshot:\n", q_table)

# Plot training rewards
plt.figure(figsize=(12, 5))
plt.plot(training_rewards, color='blue', alpha=0.4, label='Raw Rewards')
plt.plot(moving_average(training_rewards), color='green', label='Smoothed (MA)')
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Training Reward Over Time")
plt.legend()
plt.grid(True)
plt.show()

# Save Q-table
np.save("q_table_taxi.npy", q_table)

# Evaluation with rendering and success rate
env = gym.make("Taxi-v3", render_mode='human')
eval_episodes = 5
eval_rewards = []
successes = 0

for episode in range(eval_episodes):
    state, _ = env.reset()
    done = False
    total_rewards = 0

    for step in range(max_steps):
        action = np.argmax(q_table[state])
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_rewards += reward
        state = next_state

        if done:
            break

    eval_rewards.append(total_rewards)
    if total_rewards > 0:
        successes += 1
    print(f"Evaluation Episode {episode + 1}, Total Reward: {total_rewards}")

# Success rate
success_rate = successes / eval_episodes
print(f"\n✅ Success Rate: {success_rate:.2%}")
print(f"🔍 Average Evaluation Reward: {np.mean(eval_rewards):.2f}")

env.close()
