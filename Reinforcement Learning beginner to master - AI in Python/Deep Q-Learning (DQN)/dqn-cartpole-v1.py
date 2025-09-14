# =============================================================================
# DQN CART POLE BALANCING AGENT
# =============================================================================
# Task: The agent has to decide two actions - moving the cart left or right
# so that the pole attached to it stays upright.
# 
# Requirements: gymnasium[classic_control]
# 
# This implementation uses Deep Q-Network (DQN) with:
# - Experience Replay for stable learning
# - Target Network for stable Q-value estimation
# - Epsilon-greedy exploration strategy
# - Huber loss for robust training
# =============================================================================

# Task : The agent has to decide two actions - moving the cart left or right so that the pole attached to it stays upright.
 
# requirements
# gymnasium[classic_control]

"""
- Neural network (torch.nn)
- Optimization (torch.optim)
- automatic differtiation (torch.autograd)

- MPS (Metal Performance Shaders) is a framework by Apple that provides a highly optimized library of data-parallel primitives essential for machine learning, image processing, and neural network computation. It leverages the power of the GPU, not the CPU, to accelerate performance on macOS and iOS devices.
- It enables efficient GPU usage for ML workloads.
- Helps models run faster by offloading computations to the GPU.
- Not a hardware type — it is a software layer that interfaces with Apple's Metal API for GPU acceleration.

"""

# =============================================================================
# 1. ENVIRONMENT SETUP
# =============================================================================

import gymnasium as gym 
import math
import random 
import matplotlib
import matplotlib.pyplot as plt 
from collections import namedtuple, deque 
from itertools import count 

import torch 
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F

# Create the CartPole environment
# CartPole-v1: A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
# The system is controlled by applying a force of +1 or -1 to the cart.
# The pendulum starts upright, and the goal is to prevent it from falling over.
env = gym.make("CartPole-v1")

# Check if we are running in Jupyter notebook for proper display
is_python = "inline" in matplotlib.get_backend()
if is_python:
    from IPython import display

# Turn on interactive mode for real-time plotting
plt.ion()

# Device selection: Use GPU if available, otherwise CPU
# CUDA: NVIDIA GPU acceleration
# MPS: Apple Silicon GPU acceleration (M1/M2 chips)
# CPU: Fallback for all other cases
device = torch.device(
    "cuda" if torch.cuda.is_available() else 
    "mps" if torch.backends.mps.is_available() else 
    "cpu"
)

# Optional: Set random seeds for reproducible results
# Uncomment these lines if you want consistent results across runs
# This is helpful for debugging and comparing different approaches
# seed = 42
# random.seed(seed)
# torch.manual_seed(seed)
# env.reset(seed=seed)
# env.action_space.seed(seed)
# env.observation_space.seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(seed)

# =============================================================================
# 2. REPLAY MEMORY
# =============================================================================

# Define a named tuple to store experience transitions
# Each transition contains: state, action taken, next state, reward received
Transition = namedtuple("Transition",
                         ("state", "action", "new_state", "reward"))

class ReplayMemory(object):
    """
    A cyclic buffer that stores recent transitions for experience replay.
    
    Experience replay is a key technique in DQN that:
    1. Breaks correlation between consecutive experiences
    2. Allows the agent to learn from past experiences multiple times
    3. Improves sample efficiency and training stability
    """

    def __init__(self, capacity):
        """
        Initialize replay memory with specified capacity.
        
        Args:
            capacity (int): Maximum number of transitions to store
        """
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """
        Save a transition to memory.
        
        Args:
            *args: state, action, new_state, reward
        """
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """
        Randomly sample a batch of transitions for training.
        
        Args:
            batch_size (int): Number of transitions to sample
            
        Returns:
            list: Random batch of transitions
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Return current number of stored transitions."""
        return len(self.memory)
    

# =============================================================================
# 3. DQN NEURAL NETWORK
# =============================================================================

class DQN(nn.Module):
    """
    Deep Q-Network (DQN) implementation.
    
    This neural network approximates the Q-function Q(s,a) which estimates
    the expected cumulative reward for taking action 'a' in state 's'.
    
    Architecture:
    - Input layer: n_observations (4 for CartPole: position, velocity, angle, angular velocity)
    - Hidden layer 1: 128 neurons with ReLU activation
    - Hidden layer 2: 128 neurons with ReLU activation  
    - Output layer: n_actions (2 for CartPole: left, right)
    """

    # DQN Network	4 → 128 → 128 → 2 neurons
    
    def __init__(self, n_observations, n_actions):
        """
        Initialize the DQN network.
        
        Args:
            n_observations (int): Number of state features (4 for CartPole)
            n_actions (int): Number of possible actions (2 for CartPole)
        """
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input state tensor
            
        Returns:
            torch.Tensor: Q-values for each action
        """
        x = F.relu(self.layer1(x))  # First hidden layer with ReLU
        x = F.relu(self.layer2(x))  # Second hidden layer with ReLU
        return self.layer3(x)       # Output layer (no activation - raw Q-values)

# =============================================================================
# 4. TRAINING HYPERPARAMETERS AND UTILITIES
# =============================================================================

# Hyperparameters for DQN training
BATCH_SIZE = 128      # Number of transitions sampled from replay buffer
GAMMA = 0.99          # Discount factor for future rewards (0-1, closer to 1 = more future-focused)
EPS_START = 0.9       # Starting value of epsilon (exploration rate)
EPS_END = 0.01        # Final value of epsilon (minimum exploration rate)
EPS_DECAY = 2500      # Rate of epsilon decay (higher = slower decay)
TAU = 0.005           # Soft update rate for target network (0-1, closer to 1 = faster update)
LR = 3e-4             # Learning rate for Adam optimizer

# Get environment specifications
n_actions = env.action_space.n        # Number of possible actions (2 for CartPole)
state, info = env.reset()             # Reset environment to get initial state
n_observations = len(state)           # Number of state features (4 for CartPole)

# Initialize neural networks
policy_net = DQN(n_observations, n_actions).to(device)    # Main network (updated every step)
target_net = DQN(n_observations, n_actions).to(device)    # Target network (updated less frequently)
target_net.load_state_dict(policy_net.state_dict())      # Initialize target network with policy network weights

# Initialize optimizer and replay memory
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)  # AdamW optimizer with AMSGrad
memory = ReplayMemory(10000)  # Replay buffer with capacity of 10,000 transitions

steps_done = 0  # Counter for epsilon decay

def select_action(state):
    """
    Select an action using epsilon-greedy policy.
    
    Epsilon-greedy policy balances exploration vs exploitation:
    - With probability epsilon: choose random action (exploration)
    - With probability (1-epsilon): choose best action according to current policy (exploitation)
    
    Args:
        state (torch.Tensor): Current state
        
    Returns:
        torch.Tensor: Selected action
    """
    global steps_done
    sample = random.random()
    
    # Calculate current epsilon value (exponentially decaying)
    eps_threshold = EPS_END + (EPS_START-EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    
    if sample > eps_threshold:
        # Exploitation: Choose action with highest Q-value
        with torch.no_grad():
            # Get Q-values for all actions and select the one with maximum value
            return policy_net(state).max(1).indices.view(1, 1)
    else: 
        # Exploration: Choose random action
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

episode_durations = []  # Store episode lengths for plotting

def plot_durations(show_result=False):
    """
    Plot episode durations to visualize training progress.
    
    Args:
        show_result (bool): If True, show final results; if False, show training progress
    """
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
    
    # Calculate and plot 100-episode moving average for smoother visualization
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # Pause briefly to update the plot
    
    # Handle Jupyter notebook display
    if is_python:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

# =============================================================================
# 5. TRAINING OPTIMIZATION FUNCTION
# =============================================================================

def optimize_model():
    """
    Perform one step of optimization on the policy network.
    
    This function implements the core DQN training algorithm:
    1. Sample a batch of transitions from replay memory
    2. Compute current Q-values Q(s_t, a_t) using policy network
    3. Compute target Q-values using target network and Bellman equation
    4. Calculate loss between current and target Q-values
    5. Update policy network using gradient descent
    
    The target network is used for stability - it is updated less frequently
    than the policy network to prevent the moving target problem.
    """
    # Skip optimization if we do not have enough experience yet
    if len(memory) < BATCH_SIZE:
        return 
    
    # Sample a random batch of transitions from replay memory
    transitions = memory.sample(BATCH_SIZE)
    
    # Convert batch-array of Transitions to Transition of batch-arrays
    # This transposes the batch for easier processing
    batch = Transition(*zip(*transitions))

    # Create mask for non-terminal states and concatenate batch elements
    # Terminal states have next_state = None, non-terminal states have actual next states
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.new_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.new_state if s is not None])
    
    # Concatenate all batch elements into tensors
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a_t) - current Q-values for the actions that were actually taken
    # The gather function selects the Q-value for the specific action taken in each state
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states using target network
    # V(s) = max_a Q(s,a) - the maximum Q-value over all possible actions
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        # Only compute Q-values for non-terminal next states
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    
    # Compute expected Q-values using Bellman equation:
    # Q_target = r + γ * max_a Q(s', a)
    # For terminal states, Q_target = r (since there is no next state)
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss (smooth L1 loss) between current and target Q-values
    # Huber loss is less sensitive to outliers than MSE loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model using gradient descent
    optimizer.zero_grad()  # Clear gradients from previous step
    loss.backward()        # Compute gradients

    # Gradient clipping to prevent exploding gradients
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()       # Update network weights

# =============================================================================
# 6. MAIN TRAINING LOOP
# =============================================================================

# Determine number of training episodes based on available compute
if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 600  # More episodes for GPU training
else: 
    num_episodes = 500  # Fewer episodes for CPU training

print(f"Starting DQN training on {device} for {num_episodes} episodes...")
print(f"Environment: CartPole-v1")
print(f"State space: {n_observations} features")
print(f"Action space: {n_actions} actions")
print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LR}")
print("-" * 50)

'''
For each episode:
  1. Reset environment
  2. For each step:
     - Choose action (explore or exploit)
     - Take action, get reward
     - Store experience in memory
     - Train network on random batch
     - Update target network slowly
  3. Plot progress

'''

for i_episode in range(num_episodes):
    # Initialize the environment and get initial state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    
    # Run one episode until termination
    for t in count():
        # Select action using epsilon-greedy policy
        action = select_action(state)
        
        # Take action in environment and observe result
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated 

        # Determine next state
        if terminated:
            next_state = None  # Terminal state
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store transition in replay memory
        memory.push(state, action, next_state, reward)

        # Move to next state
        state = next_state 

        # Perform one step of optimization on policy network
        optimize_model()

        # Soft update of target network weights
        # θ_target ← τ * θ_policy + (1-τ) * θ_target
        # This slowly updates the target network towards the policy network
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1-TAU) 
        target_net.load_state_dict(target_net_state_dict)

        # End episode if done
        if done:
            episode_durations.append(t+1)
            plot_durations()
            
            # Print progress every 50 episodes
            if i_episode % 50 == 0:
                avg_duration = sum(episode_durations[-50:]) / min(50, len(episode_durations))
                print(f"Episode {i_episode}, Duration: {t+1}, Avg (last 50): {avg_duration:.1f}")
            break

print("\\nTraining complete!")
print(f"Average duration over last 100 episodes: {sum(episode_durations[-100:]) / min(100, len(episode_durations)):.1f}")
plot_durations(show_result=True)
plt.ioff()
plt.show()
