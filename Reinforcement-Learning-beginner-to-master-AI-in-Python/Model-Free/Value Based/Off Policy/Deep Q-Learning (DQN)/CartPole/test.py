import gymnasium as gym
import torch
from itertools import count
from dqn_cartpole import DQN, load_model, device

def test_agent(episodes=5, render=True, model_path="models/dqn_cartpole_solved.pth"):
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    state, _ = env.reset()
    n_observations = len(state)
    n_actions = env.action_space.n

    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    optimizer = torch.optim.AdamW(policy_net.parameters())

    load_model(model_path, policy_net, target_net, optimizer)

    for episode in range(episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            with torch.no_grad():
                action = policy_net(state).max(1).indices.view(1, 1)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            if done:
                print(f"Episode {episode+1} finished after {t+1} steps")
                break
            state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    env.close()

if __name__ == "__main__":
    test_agent()
