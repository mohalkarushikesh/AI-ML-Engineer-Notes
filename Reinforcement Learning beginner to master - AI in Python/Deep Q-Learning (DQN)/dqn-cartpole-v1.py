# Imports
import gymnasium as gym
import math, random, os
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Environment setup
env = gym.make("CartPole-v1")
is_python = "inline" in matplotlib.get_backend()
if is_python:
    from IPython import display
plt.ion()

device = torch.device(
    "cuda" if torch.cuda.is_available() else 
    "mps" if torch.backends.mps.is_available() else 
    "cpu"
)

# Replay memory
Transition = namedtuple("Transition", ("state", "action", "new_state", "reward"))
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args):
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

# DQN model
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# Hyperparameters
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 2500
TAU = 0.005
LR = 3e-4

# Initialization
n_actions = env.action_space.n
state, info = env.reset()
n_observations = len(state)
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)
steps_done = 0
episode_durations = []

# Action selection
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

# Plotting
def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title("Result")
    else:
        plt.clf()
        plt.title("Training")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(durations_t.numpy())
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    plt.pause(0.001)
    if is_python:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

# Optimization
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
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
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

# Model saving/loading
def save_model(policy_net, target_net, optimizer, episode, avg_duration, filepath=None):
    os.makedirs("models", exist_ok=True)
    if filepath is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"models/dqn_cartpole_{timestamp}.pth"
    checkpoint = {
        'episode': episode,
        'policy_net_state_dict': policy_net.state_dict(),
        'target_net_state_dict': target_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'avg_duration': avg_duration,
        'episode_durations': episode_durations,
        'steps_done': steps_done,
        'hyperparameters': {
            'BATCH_SIZE': BATCH_SIZE,
            'GAMMA': GAMMA,
            'EPS_START': EPS_START,
            'EPS_END': EPS_END,
            'EPS_DECAY': EPS_DECAY,
            'TAU': TAU,
            'LR': LR,
            'n_observations': n_observations,
            'n_actions': n_actions
        }
    }
    torch.save(checkpoint, filepath)
    print(f"Model saved to: {filepath}")
    return filepath

def load_model(filepath, policy_net, target_net, optimizer):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    target_net.load_state_dict(checkpoint['target_net_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    global steps_done, episode_durations
    steps_done = checkpoint['steps_done']
    episode_durations = checkpoint['episode_durations']
    print(f"Model loaded from: {filepath}")
    print(f"Resuming from episode: {checkpoint['episode']}")
    print(f"Previous avg duration: {checkpoint['avg_duration']:.1f}")
    return checkpoint

def save_checkpoint(policy_net, target_net, optimizer, episode, avg_duration):
    os.makedirs("models/checkpoints", exist_ok=True)
    filepath = f"models/checkpoints/checkpoint_episode_{episode}.pth"
    return save_model(policy_net, target_net, optimizer, episode, avg_duration, filepath)

def list_saved_models():
    models = []
    if os.path.exists("models"):
        for file in os.listdir("models"):
            if file.endswith('.pth'):
                models.append(os.path.join("models", file))
    return sorted(models)

# Training loop
num_episodes = 600 if torch.cuda.is_available() or torch.backends.mps.is_available() else 500
print(f"Starting DQN training on {device} for {num_episodes} episodes...")
print("-" * 50)

'''
for i_episode in range(num_episodes):
    state, info = env.reset()
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

        # Soft update of target network
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            avg_duration = sum(episode_durations[-100:]) / min(100, len(episode_durations))

            if i_episode % 50 == 0:
                print(f"Episode {i_episode}, Duration: {t+1}, Avg (last 100): {avg_duration:.1f}")
                save_checkpoint(policy_net, target_net, optimizer, i_episode, avg_duration)

            if avg_duration >= 195 and i_episode >= 100:
                print(f"\n🎉 SOLVED! Average duration: {avg_duration:.1f}")
                save_model(policy_net, target_net, optimizer, i_episode, avg_duration, "models/dqn_cartpole_solved.pth")
                break

            break  # end of episode

# Final model save
final_avg = sum(episode_durations[-100:]) / min(100, len(episode_durations))
save_model(policy_net, target_net, optimizer, num_episodes, final_avg, "models/dqn_cartpole_final.pth")
print("\nTraining complete!")
print(f"Average duration over last 100 episodes: {final_avg:.1f}")
plot_durations(show_result=True)
plt.ioff()
plt.show()
'''

def test_agent(episodes=5, render=True, model_path=None):
    if model_path and os.path.exists(model_path):
        print(f"Loading model from: {model_path}")
        load_model(model_path, policy_net, target_net, optimizer)
    test_env = gym.make("CartPole-v1", render_mode="human" if render else None)
    print(f"\nTesting trained agent for {episodes} episodes...")
    print("Close the rendering window to continue...")
    total_steps = 0
    successful_episodes = 0
    for episode in range(episodes):
        state, info = test_env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        total_reward = 0
        for step in count():
            with torch.no_grad():
                action = policy_net(state).max(1).indices.view(1, 1)
            observation, reward, terminated, truncated, _ = test_env.step(action.item())
            total_reward += reward
            done = terminated or truncated
            if done:
                total_steps += step + 1
                if step + 1 >= 195:
                    successful_episodes += 1
                print(f"Episode {episode + 1}: {step + 1} steps, Total Reward: {total_reward:.1f}")
                break
            state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    test_env.close()
    avg_steps = total_steps / episodes
    success_rate = (successful_episodes / episodes) * 100
    print(f"\nTest Results:")
    print(f"Average steps per episode: {avg_steps:.1f}")
    print(f"Success rate (≥195 steps): {success_rate:.1f}%")
    print(f"Successful episodes: {successful_episodes}/{episodes}")

def render_training_episode(episode_num):
    render_env = gym.make("CartPole-v1", render_mode="human")
    state, info = render_env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    print(f"\nRendering Episode {episode_num}...")
    print("Close the rendering window to continue training...")
    for step in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = render_env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated
        if done:
            print(f"Rendered episode completed in {step + 1} steps")
            break
        state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    render_env.close()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("DQN CART POLE - TESTING WITH RENDERING")
    print("="*60)
    saved_models = list_saved_models()
    if saved_models:
        print("Available saved models:")
        for i, model in enumerate(saved_models):
            print(f"  {i+1}. {model}")
        print()
    test_agent(episodes=3, render=True)
    solved_model_path = "models/dqn_cartpole_final.pth"
    if os.path.exists(solved_model_path):
        print(f"\nTesting with solved model: {solved_model_path}")
        test_agent(episodes=5, render=True, model_path=solved_model_path)
    else:
        print(f"\nSolved model not found: {solved_model_path}")
        print("This is normal if the agent hasn't solved the environment yet.")
    print("\nInteractive testing - Press Enter to test again or 'q' to quit...")
    while True:
        user_input = input()
        if user_input.lower() == 'q':
            break
        test_agent(episodes=1, render=True)
