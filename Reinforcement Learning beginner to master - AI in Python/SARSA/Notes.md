SARSA (state-action-reward-state-action) is an on-policy reinforcement learning algorithm that helps an agent to learn optimal policy by interacting with it's enviroment
The agent explores it's environment, takes actions, receives feedback and continuously updates it's behaviour to maximize long-term rewards.

sarsa updates it's Q-Value using bellman equation for sarsa

Sure! Here's the Q-learning update rule written clearly:

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t) \right]
$$

Where:
- $Q(s_t, a_t)$: current Q-value for state $s_t$ and action $a_t$
- $\alpha$: learning rate
- $r_{t+1}$: reward received after taking action $a_t$
- $\gamma$: discount factor
- $Q(s_{t+1}, a_{t+1})$: estimated Q-value for next state-action pair

This rule adjusts the Q-value toward the target value based on the reward and expected future value.

Breaking down the update rule

- Immediate reward : The agent receives an immediate reward $r_{t+1}$ after taking the action $a_t$ in state $s_t$.
- Future reward : The expected future reward is calculated as $Q(s_t+1, a_t+1)$, the Q-Value of the next state action pair
- Correction : The agent update the Q-Value for current state action pair based on the difference between the predicted reward and actual reward received.

This update rule allows agent to increase it's policy incrementaly, improving decision-making over time.

SARSA Algorithm steps
1. Initialize Q-Value : arbitary
2. Choose a state
3. Episode loop
4. Step loop
5. End condition

Implementing the SARSA Algorithm
1. Defining the environment (GridWorld)
2. Defining the sarsa algorithm
3. Defning the epsilon greedy policy
4. Setup the environment and running the sarsa

Exploration strategies for sarsa
1. Exploration : with probability of the epsilon, the agent choose the random action (exploring the new posibilities)
2. Exploitation : with probabililty of the 1 - epsilon, the agent chooses hight q-value for current state (exploiting it's current knoweledge)

Advantages :
1. On-policy learning: where the exploration and behaviour directly influence learning
2. Real-world behaviour : The agent learns from the real experiences, leading to the grounded decision making that reflects it's actual behaviour in uncertain situations.
3. Gradual improvement : it is more stable than off-policy methods like Q-Learning.

Limitations :
1. Slower convergence
2. Sensitive to exploration strategy : it's performance highly dependent on exploration strategy, improper learning can delay or hinder the learning.
