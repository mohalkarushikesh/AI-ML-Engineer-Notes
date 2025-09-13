Q-Learning 
- model-free reinforcement learning algorithm 
- Q-learning is an **off-policy TD control algorithm**. It uses the Bellman optimality equation to update Q-values:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]
$$	
- $\alpha$: Learning rate
- The term inside the brackets is the **TD error** — the difference between the predicted and actual value.

Key components 
1. Q-Values or Action-Values 
2. Rewards and Episodes 
3. Temporal Difference or TD-Update
4. E-greedy Policy (Exploration vs. Exploitation)
a. Exploration : The agent picks the action with highest Q-value with probability 1-E. this means agent uses it's current knowledge ot maximize rewards 
b. Exploitation : With probability E, the agent picks a random action, exploring new possibilities to learn if there are better ways to get rewards. This allows the agent to discover new strategies and improve it's decision-making over time 

working 
1. Initialize Q table
2. Choose an action 
3. perform action 
4. Measure reward 
5. Update the Q-table 

Methods for Determining Q-values 
1. Temporal Difference : is calculated by comparing current state and action values of prev ones. It provides a way to learn directly from experience, without needing a model of the enviroment.
TD learning blends ideas from Monte Carlo and Dynamic Programming. The TD error is:

$$
\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)
$$

Or for Q-values:

$$
\delta_t = r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)
$$

This error measures how far off the current estimate is from the observed reward plus the estimated future value.
  
2. Bellaman's Equation : is a recurssive formula used to calculate with given state and determine the optimal action. It is fundamental in the context of Q-Learning.

This is the foundation of value-based reinforcement learning. It expresses the expected value of a state-action pair:

$$
Q(s, a) = \mathbb{E} \left[ r + \gamma \max_{a'} Q(s', a') \mid s, a \right]
$$

- $Q(s, a)$: Value of taking action $a$ in state $s$
- $r$: Immediate reward
- $\gamma$: Discount factor (how much future rewards matter)
- $s'$: Next state
- $a'$: Next possible actions

This equation is **recursive** — it defines the value of a state-action pair in terms of the values of future state-action pairs.


Q-table 
	- Row represent state
	- column represent action
	- each entry in the table corresponds to the Q-value for state-action pair.

Ex
1. Define the environments 
2. Set the hyper-parameters 
3. Implement the Q-Learning Algorithm
4. Output the Learned Q-Table 

Advatages 
1. Trial and Error learning
2. Self-Improvement
3. Better-Decision making 
4. Autonomous learning

Dis-Advantages 
1. Slow learning 
2. Expensive in some enviroments
3. Curse of dimentionality
4. Limited to Discete actions
