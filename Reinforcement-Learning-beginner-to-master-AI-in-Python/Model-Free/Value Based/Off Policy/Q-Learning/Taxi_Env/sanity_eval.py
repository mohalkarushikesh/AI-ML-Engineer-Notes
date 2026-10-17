import numpy as np

q_table = np.load("q_table_taxi.npy")
print("Shape:", q_table.shape)          # expect (500, 6) for Taxi-v3
print("Dtype:", q_table.dtype)
print("Any NaNs?", np.isnan(q_table).any())
print("All zeros?", np.all(q_table == 0))
print("Min/Max:", q_table.min(), q_table.max())
