import gymnasium as gym
from itertools import count
import random
import math
import torch
import torch.optim as optim
from dqn_cartpole import DQN, ReplayMemory, plot_durations, save_model, load_model, device

# Hyperparameters
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 2500
TAU = 0.005
LR = 3e-4

env = gym.make("CartPole-v1")
n_actions = env.action_space.n
state, _ = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0
episode_durations = []

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = ReplayMemory.Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.new_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.new_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    loss = torch.nn.SmoothL1Loss()(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

# Training loop
num_episodes = 500
for i_episode in range(num_episodes):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated
        next_state = None if terminated else torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        memory.push(state, action, next_state, reward)
        state = next_state
        optimize_model()
        for key in policy_net.state_dict():
            target_net.state_dict()[key] = policy_net.state_dict()[key] * TAU + target_net.state_dict()[key] * (1 - TAU)
        if done:
            episode_durations.append(t + 1)
            plot_durations(episode_durations)
            avg_duration = sum(episode_durations[-100:]) / min(100, len(episode_durations))
            if i_episode % 50 == 0:
                save_model(policy_net, target_net, optimizer, i_episode, avg_duration, f"models/checkpoint_{i_episode}.pth")
            if avg_duration >= 195 and i_episode >= 100:
                save_model(policy_net, target_net, optimizer, i_episode, avg_duration, "models/dqn_cartpole_solved.pth")
                break
            break

final_avg = sum(episode_durations[-100:]) / min(100, len(episode_durations))
save_model(policy_net, target_net, optimizer, num_episodes, final_avg, "models/dqn_cartpole_final.pth")
plot_durations(episode_durations, show_result=True)
