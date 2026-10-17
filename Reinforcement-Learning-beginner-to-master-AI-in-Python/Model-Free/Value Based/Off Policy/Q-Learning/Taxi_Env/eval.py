import gymnasium as gym
import numpy as np

q_table = np.load("q_table_taxi.npy")
env = gym.make("Taxi-v3", render_mode="human")

n_episodes = 100
total_rewards = []
successes = 0

for ep in range(n_episodes):
    state, _ = env.reset()
    done = False
    ep_reward = 0
    steps = 0
    while not done and steps < 200:
        action = np.argmax(q_table[state])   # greedy policy
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        ep_reward += reward
        steps += 1
    total_rewards.append(ep_reward)
    if ep_reward > 0:   # successful drop-off gives +20, so positive reward = success
        successes += 1

print(f"Avg reward over {n_episodes} episodes: {np.mean(total_rewards):.2f}")
print(f"Success rate: {successes/n_episodes*100:.1f}%")
