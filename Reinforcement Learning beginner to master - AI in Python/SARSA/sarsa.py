# Step 1 : Define the environment (GridWorld)

import numpy as np 
import random 

class GridWorld: 
    def __init__(self, width, height, start, goal, obstacles): 
        self.width = width
        self.height = height
        self.start = start 
        self.goal = goal
        self.obstacles = obstacles 
        self.state = start             # sets the agent's initial position 
    
    def reset(self):                   # reset the env to it's initial state (starting of the new episodes)
        self.state = self.start 
        return self.state           

    def step(self, action):            # Ensure agent stay within the grid boundaries using max and min 
        x, y = self.state 
        if action == 0: 
            x = max(x-1, 0)
        elif action == 1: 
            x = min(x+1, self.height-1)
        elif action == 2: 
            y = max(y-1, 0)
        elif action == 3: 
            y = min(y+1, self.width-1)  

        if (x, y) in self.obstacles:  
            reward = -10
            done = True
        elif (x, y) == self.goal:  
            reward = 10 
            done = True 
        else: 
            reward = -1
            done = False  
    
        self.state = (x, y)   
        return (x, y), reward, done  
        

# Step 2 : Define the sarsa algorithm 
def sarsa(env, episodes, alpha, gamma, epsilon):
    Q = np.zeros((env.height, env.width, 4))
    
    for episode in range(episodes):
        state = env.reset()
        action = epsilon_greedy_policy(Q, state, epsilon)                   # Choose an action based on epsilon greedy policy (explore or exploit )
        done = False 

        while not done: 
            next_state, reward, done = env.step(action)  
            next_action  = epsilon_greedy_policy(Q, next_state, epsilon)    # Choose an action based on epsilon greedy policy again 

            # target = reward + gamma * Q(next_state, next_action)
            Q[state[0], state[1], action] += alpha * (reward + gamma * Q[next_state[0], next_state[1], next_action] - Q[state[0], state[1], action])
        
            state = next_state
            action = next_action 

    return Q

# Step 3 : Define epsilon greedy policy 
def epsilon_greedy_policy(Q, state, epsilon):  
    if random.uniform(0, 1) < epsilon:         # with the probability of epsilon chooses the action 0 to 3
        return random.randint(0, 3)
    else:                                      # otherwise choose the action with highest Q-value for the current state 
        return np.argmax(Q[state[0], state[1]])

# Step 4 : Setup the environment and execute the sarsa 
if __name__ == "__main__":  # 

    width = 5
    height = 5
    start = (0, 0)
    goal = (4, 4)
    obstacles = [(2, 2), (3, 2)]
    env = GridWorld(width, height, start, goal, obstacles)

    episodes = 1000
    alpha = 0.1 
    gamma = 0.99
    epsilon = 0.1

    Q = sarsa(env, episodes, alpha, gamma, epsilon)

    print("Learned Q-Values: ")
    print(Q)
  
```
Learned Q-Values:
[[[-1.1218239  -0.79153618 -0.954389    0.55400138]
  [-0.219239    1.84266886 -1.2861059   0.84730617]
  [-1.87187937  2.99642356 -1.36977919 -1.57745576]
  [-0.96079413  4.00052598 -1.23248769 -1.36827025]
  [-0.63865903  5.14499143 -1.00957684 -0.97705848]]

 [[-2.90755398 -2.94074933 -2.59659899  1.3560311 ]
  [ 0.08503364 -0.39924604 -1.98887576  2.99349206]
  [ 0.74908433 -9.57608842  0.9836226   4.71150079]
  [ 1.27662832  1.92419766  2.67009062  6.10518225]
  [ 1.84040814  7.38984424  3.44983486  4.75960598]]

 [[-2.32775001 -2.32554251 -2.35834643 -2.37536584]
  [ 1.53142242 -1.95227424 -2.11179865 -4.0951    ]
  [ 0.          0.          0.          0.        ]
  [-0.47120619  6.28798397 -1.9         0.62132346]
  [ 4.32685206  8.57916693  2.20291522  7.01681866]]

 [[-1.73920637 -1.76336571 -1.83525687 -1.7631314 ]
  [-1.47016449 -0.58088654 -1.37109383 -1.9       ]
  [ 0.          0.          0.          0.        ]
  [ 0.24567635  0.72103122 -1.9         8.42767834]
  [ 6.72566333 10.          3.79848455  7.68959905]]

 [[-1.31933343 -1.33074379 -1.35689128 -1.07357501]
  [-0.9749992  -1.04339749 -0.86736519  2.20495362]
  [-3.439      -0.55412033 -0.56254523  5.95940495]
  [ 0.05083428  0.46735528 -0.33341732  9.35389181]
  [ 0.          0.          0.          0.        ]]]

```
