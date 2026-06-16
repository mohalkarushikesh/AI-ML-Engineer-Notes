## 🧠 Long Short-Term Memory (LSTM)

LSTM is a specialized RNN architecture designed to **remember long-term dependencies**.

### 🧬 LSTM Cell Components
- **x**: Current input  
- **h**: Previous **hidden state** (Short-Term Memory)
- **c**: Cell state (Long-Term Memory)  
- **Gates**:
  - **Forget Gate**: Decides what to discard  
  - **Input Gate**: Decides what to store  
  - **Output Gate**: Decides what to pass forward  

<img width="723" height="477" alt="A-Long-short-term-memory-LSTM-unit-architecture" src="https://github.com/user-attachments/assets/8a629f96-d4a2-4eda-83c5-4a6850c5d29b" />

### 🧪 Pointwise Operations
Each gate uses element-wise operations to control the flow of information.

---

## 📚 Foundational Papers & Further Reading

- *Untersuchungen zu dynamischen* – Sepp Hochreiter (1991)  
- *Learning Long-Term Dependencies* – Yoshua Bengio et al. (1994)  
- *On the Difficulty of Training RNNs* – Razvan Pascanu et al. (2013)  
- *Understanding LSTM Networks* – [Christopher Olah](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)  
- *Understanding LSTM Diagrams* – [Shi Yan](https://blog.mlreview.com/understanding-lstm-and-its-diagrams-37e2f46f1714)

---

## 📚 How LSTM Works & Visualization

- [*The Unreasonable Effectiveness of Recurrent Neural Networks* – Andrej Karpathy (2015)](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)  
- [*Visualizing and Understanding Recurrent Networks* – Andrej Karpathy et al. (2015)](https://arxiv.org/abs/1506.02078)  
- [*LSTM: A Search Space Odyssey* – Klaus Greff et al. (2015)](https://arxiv.org/abs/1503.04069)  

---

## 🧩 LSTM Variations

- **Bidirectional LSTM**: Processes input in both directions  
- **Stacked LSTM**: Multiple LSTM layers for deeper learning  
- **CNN-LSTM**: Combines convolutional and sequential processing  
- **Attention-based LSTM**: Focuses on relevant parts of the sequence  

---

## 📊 Feature Scaling

### 🔹 Min-Max Scaling
Transforms feature values to a fixed range (usually [0, 1]).  

$$
X_{scaled} = \frac{X - X_{min}}{X_{max} - X_{min}}
$$  

✅ Benefits: Makes training more stable & prevents feature dominance.

---

### 🔹 Standardization (Z-score Normalization)
Centers data around mean 0 and standard deviation 1.  

$$
z = \frac{x - \mu}{\sigma}
$$  

Where:  
- $x$ = original value  
- $\mu$ = mean of the feature  
- $\sigma$ = standard deviation  

<img width="1000" height="374" alt="image" src="https://github.com/user-attachments/assets/32884409-e310-4ae0-b05f-ea1c7947f8b1" />

# Standard Deviation

**Standard deviation** is a statistical measure that quantifies the amount of variation or dispersion of a set of data points around their average (mean). A **low standard deviation** indicates that data points cluster tightly around the mean, while a **high standard deviation** indicates wider, more spread-out values.

## Why Standard Deviation Matters

- **Direct Scaling:** Unlike variance (which measures squared deviations), standard deviation returns to the original units of your data, making it easier to interpret and apply.

- **Risk Assessment:** In finance, it acts as a primary benchmark for asset volatility.

- **The 68-95-99.7 Rule:** In a standard bell curve:
  - Approximately **68%** of data falls within one standard deviation of the mean.
  - Approximately **95%** falls within two standard deviations.
  - Approximately **99.7%** falls within three standard deviations.

## How It's Calculated

Standard deviation (\(\sigma\)) is mathematically the **square root of the variance**. Calculating it manually involves the following steps:

1. Calculate the **average (mean)** of the dataset.
2. Subtract the mean from each individual data point and square the result to remove negative values.
3. Find the average of those squared results (this is the **variance**).
4. Take the square root of the variance.

## Formula

For a population:

$\sigma = \sqrt{\frac{\sum_{i=1}^{N}(x_i-\mu)^2}{N}}$

Where:

- $sigma$ = Population standard deviation
  $x_i$ = Individual data point
- $mu$ = Population mean
- $N$ = Number of data points

For a sample:

$s = \sqrt{\frac{\sum_{i=1}^{n}(x_i-\bar{x})^2}{n-1}}$

Where:

- $s$ = Sample standard deviation
- $x_i$ = Individual data point
- $bar{x}$ = Sample mean
- $n$ = Sample size

## Practical Uses

Standard deviation is widely used across several disciplines:

- **Finance:** Measures an investment's risk; higher deviations correspond to greater price fluctuations.

- **Healthcare and Science:** Helps researchers define a "normal" range (e.g., blood pressure or height) and determine statistical significance.

- **Quality Control:** Used to monitor manufacturing processes and maintain consistency.

- **Data Analysis and Machine Learning:** Assesses variability and aids in feature scaling and anomaly detection.

## Key Takeaways

- Standard deviation measures how spread out data points are around the mean.
- A lower value indicates less variability, while a higher value indicates greater variability.
- It is the square root of variance, making it easier to interpret because it uses the same units as the original data.
- The **68-95-99.7 rule** provides a quick way to understand distributions that follow a normal curve.
---

### 🔹 Normalization (Min-Max Normalization)
Rescales values to [0, 1].  

$$
x' = \frac{x - x_{min}}{x_{max} - x_{min}}
$$  

Where:  
- $x$ = original value  
- $x_{min}$ = minimum value of the feature  
- $x_{max}$ = maximum value of the feature  
- $x'$ = normalized value (between 0 and 1)

---
