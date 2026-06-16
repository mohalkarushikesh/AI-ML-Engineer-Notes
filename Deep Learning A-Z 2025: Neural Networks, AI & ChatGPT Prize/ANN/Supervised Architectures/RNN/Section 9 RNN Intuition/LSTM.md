## 🧠 Long Short-Term Memory (LSTM)

LSTM is a specialized RNN architecture designed to **remember long-term dependencies**.

### 🧬 LSTM Cell Components
- **x**: Current input  
- **h**: Previous hidden state  
- **c**: Cell state (memory)  
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

<img width="850" height="300" alt="image" src="https://github.com/user-attachments/assets/4c420a32-8426-4db3-8060-a0b0337e332f" />

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
