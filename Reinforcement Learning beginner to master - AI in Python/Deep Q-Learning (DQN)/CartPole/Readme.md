# DQN CartPole - Modular Implementation

##  Project Structure

```
DQN/
 config.py              # Configuration and hyperparameters
 memory.py              # Replay memory implementation
 network.py             # DQN neural network
 agent.py               # DQN agent with training logic
 main.py                # Training script
 test.py                # Testing script
 dqn-cartpole-v1.py     # Original monolithic file (backup)
```

##  Quick Start

### Training
```bash
python main.py
```

### Testing
```bash
python test.py
```

##  Module Descriptions

### config.py
- All hyperparameters and settings
- Device configuration (CUDA/MPS/CPU)
- Model saving configuration
- Environment settings

### memory.py
- ReplayMemory class for experience replay
- Transition namedtuple for storing experiences
- Batch sampling functionality

### network.py
- DQN neural network implementation
- Forward pass and action selection
- Configurable hidden layer sizes

### agent.py
- DQNAgent class with all training logic
- Epsilon-greedy action selection
- Model optimization and target network updates
- Memory management

### main.py
- Training script
- Environment setup
- Training loop
- Progress monitoring

### test.py
- Testing script
- Visual rendering
- Performance evaluation

##  Usage Examples

### Basic Training
```python
from agent import DQNAgent
from config import *

device = get_device()
agent = DQNAgent(n_observations=4, n_actions=2, device=device)
# Training logic...
```

### Testing with Rendering
```python
from test import test_agent
test_agent(episodes=5, render=True)
```

##  Configuration

Edit `config.py` to modify:
- Hyperparameters (learning rate, batch size, etc.)
- Network architecture
- Training episodes
- Model saving settings
- Device preferences

##  Benefits of Modular Structure

1. **Separation of Concerns**: Each module has a specific responsibility
2. **Reusability**: Components can be used independently
3. **Maintainability**: Easy to modify individual components
4. **Testability**: Each module can be tested separately
5. **Scalability**: Easy to add new features or environments
