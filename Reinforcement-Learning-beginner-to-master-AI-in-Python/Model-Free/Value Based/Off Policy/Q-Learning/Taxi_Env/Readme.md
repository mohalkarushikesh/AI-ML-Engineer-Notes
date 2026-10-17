## Taxi-v3 Q-Learning

This README documents a tabular Q-Learning agent trained on `Taxi-v3` using `gymnasium`. The agent learns to navigate the grid world, pick up a passenger, and drop them off at the destination.

### Dependencies

- `gymnasium`
- `numpy`
- `matplotlib`

Install (example):
```
pip install gymnasium numpy matplotlib
```

### How to Run

```
python taxi.py
```

What it does:
- Trains a Q-table for up to 10,000 episodes with ε-greedy exploration and learning-rate decay.
- Prints periodic training logs (moving average of last 100 rewards).
- Early-stops when the agent performs consistently well.
- Saves `q_table_taxi.npy` and shows a reward plot.
- Runs 5 evaluation episodes with `render_mode='human'` and prints success metrics.

### Hyperparameters (from `taxi.py`)
- episodes: 10,000 (early stopping enabled)
- alpha (learning rate): 0.9 with decay to min 0.1
- gamma (discount): 0.95
- epsilon (exploration): 1.0 → 0.01 (decay 0.9995)
- max_steps per episode: 100
- early stopping criterion: mean of last 100 training rewards > 8


### Sanity Evaluation 
```
Shape: (500, 6)
Dtype: float64
Any NaNs? False
All zeros? False
Min/Max: -18.169783011133653 20.0
```

### Training Logs (sample)
```
Episode 0, Epsilon: 1.000, Avg Reward (last 100): -415.00
Episode 500, Epsilon: 0.778, Avg Reward (last 100): -270.90
Episode 1000, Epsilon: 0.606, Avg Reward (last 100): -74.18
Episode 1500, Epsilon: 0.472, Avg Reward (last 100): -46.63
Episode 2000, Epsilon: 0.368, Avg Reward (last 100): -21.94
Episode 2500, Epsilon: 0.286, Avg Reward (last 100): -16.89
Episode 3000, Epsilon: 0.223, Avg Reward (last 100): -8.65
Episode 3500, Epsilon: 0.174, Avg Reward (last 100): -5.20
Episode 4000, Epsilon: 0.135, Avg Reward (last 100): -0.28
Episode 4500, Epsilon: 0.105, Avg Reward (last 100): 2.12
Episode 5000, Epsilon: 0.082, Avg Reward (last 100): 2.65
Episode 5500, Epsilon: 0.064, Avg Reward (last 100): 4.54
Episode 6000, Epsilon: 0.050, Avg Reward (last 100): 5.33
Episode 6500, Epsilon: 0.039, Avg Reward (last 100): 5.14
Episode 7000, Epsilon: 0.030, Avg Reward (last 100): 6.49
Episode 7500, Epsilon: 0.023, Avg Reward (last 100): 6.25
Episode 8000, Epsilon: 0.018, Avg Reward (last 100): 7.34
Early stopping: agent consistently performs well.

Training complete!
```

### Final Q-table Snapshot (excerpt)
```
[[ 0.          0.          0.          0.          0.          0.        ]
 [ 2.75200091  3.94931301  2.75190934  3.94944272  5.20997639 -5.05052795]
 [ 7.93349182  9.40367552  7.93349183  9.40367562 10.9512375   0.40367562]
 ...
 [10.94501043 12.58024968 10.73586818  9.39813411  1.9217334   1.77586872]
 [ 4.94292651  3.4427831   5.07670138  6.53681725 -3.84616406 -3.86646917]
 [16.09999946 14.2949999  16.09999981 18.          7.1         7.1       ]]
```

### Evaluation Results
```
Evaluation Episode 1, Total Reward: 13
Evaluation Episode 2, Total Reward: 7
Evaluation Episode 3, Total Reward: 6
Evaluation Episode 4, Total Reward: 6
Evaluation Episode 5, Total Reward: 7

✅ Success Rate: 100.00%
🔍 Average Evaluation Reward: 7.80
```

### Test result

<img width="1200" height="500" alt="Training reward over time" src="https://github.com/user-attachments/assets/8156d682-acbc-44ad-b2c6-fb4d5a2e71e3" />

### Notes
- The moving average plot helps visualize training stability (saved via `matplotlib`).
- `q_table_taxi.npy` stores the resulting Q-table for reuse.
- Rendering during evaluation uses `env = gym.make("Taxi-v3", render_mode='human')`.
- Results are stochastic and may vary across runs due to exploration and environment randomness.
