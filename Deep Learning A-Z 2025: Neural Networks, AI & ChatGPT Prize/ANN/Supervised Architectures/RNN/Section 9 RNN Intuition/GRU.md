# Gated Recurrent Unit (GRU)

A **Gated Recurrent Unit (GRU)** is a streamlined Recurrent Neural Network (RNN) architecture designed to process sequential data. Introduced in **2014 by Kyunghyun Cho et al.**, GRUs solve the traditional RNN **vanishing gradient problem** by using gating mechanisms to selectively retain or discard information over long sequences.

## How GRUs Work

Instead of the complex multi-cell state used in older **Long Short-Term Memory (LSTM)** models, GRUs consolidate memory into a **single hidden state**. They control information flow using two primary gates:

- **Update Gate:** Decides how much past information (from previous time steps) needs to be passed down to the future. This is vital for maintaining long-term dependencies.

- **Reset Gate:** Decides how much of the past information to forget. This allows the model to drop irrelevant data and focus on new inputs.

<img width="850" height="405" alt="image" src="https://github.com/user-attachments/assets/5918d1bf-939b-4a96-b565-9eb68de1a500" />

<img width="1384" height="533" alt="image" src="https://github.com/user-attachments/assets/c218c5fc-9d1d-49ec-a06f-4877d1d8dc5f" />

## GRU vs. LSTM

| Feature | LSTM | GRU |
|-----------|------|-----|
| **Architecture** | More complex (Separate cell state and hidden state) | Simpler (Only a hidden state) |
| **Parameters** | Higher (3 gates: Input, Output, Forget) | Lower (2 gates: Update, Reset) |
| **Training Speed** | Slower | Faster |
| **Performance** | Better for exceptionally long sequences | Performs similarly to LSTM on many tasks |

## Common Use Cases

Because of their efficiency, GRUs are widely adopted across several AI and data domains:

- **Time Series Forecasting:** Predicting stock prices, temperature changes, and load forecasting.
- **Natural Language Processing (NLP):** Language translation, text generation, and sentiment analysis.
- **Speech and Audio:** Speech recognition and audio synthesis.

## Implementation Example (Python/TensorFlow)

Implementing a GRU in Python is straightforward using modern machine learning libraries like **TensorFlow** or **PyTorch**.

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense

# Define a Sequential GRU model
model = Sequential([
    GRU(64, input_shape=(timesteps, features)),
    Dense(1)
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='mean_squared_error'
)

# Display model summary
model.summary()
```

## Key Advantages of GRUs

- Simpler architecture compared to LSTMs.
- Fewer parameters, resulting in faster training.
- Effective at capturing long-term dependencies.
- Comparable performance to LSTMs on many sequence modeling tasks.
- Suitable for applications with limited computational resources.
