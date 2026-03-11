# **Recurrent Neural Networks (RNNs)**

RNNs are designed for **sequential data** (where the order matters, like text, audio, or stock prices). Unlike standard neural networks, they have loops that allow information to persist.

---

## 1. The Core Concept: Hidden State

In a traditional neural network, inputs are independent. In an RNN, the network takes the current input () **and** the output from the previous step ().

The formula for the hidden state is:


* : Current hidden state (the "memory").
* : Activation function (usually  or ).
* : Weight matrices.

---

![nfa-](https://github.com/user-attachments/assets/caec8b16-1e07-4fab-b6a6-cc4d3362c6e1)

---

## 2. RNN Architectures

Depending on the input and output, RNNs can be mapped differently:

| Type | Example |
| --- | --- |
| **One-to-One** | 
| **One-to-Many** | Image Captioning (One image  sequence of words). |
| **Many-to-One** | Sentiment Analysis (Sequence of words  one star rating). |
| **Many-to-Many** | Machine Translation (English sentence  French sentence). |

---

<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/66ae7023-61bd-4323-841c-f7dc24b8f855" />

---

### Weight calculation 

<img width="1312" height="512" alt="image" src="https://github.com/user-attachments/assets/0f97af8f-4de6-475e-9ca1-e6998a26c449" />


### Forward Propogation: 

<img width="1199" height="567" alt="image" src="https://github.com/user-attachments/assets/5ac82aa8-f76b-4290-9e85-159ea305eafe" />
<img width="1108" height="264" alt="image" src="https://github.com/user-attachments/assets/99a0d7bc-0c6e-4b7b-bf0f-262d636d36ad" />

### Backword propogation 

<img width="1319" height="624" alt="image" src="https://github.com/user-attachments/assets/72b1f83c-da5e-4d31-b664-91347f9dcd59" />
<img width="1351" height="646" alt="image" src="https://github.com/user-attachments/assets/dae133bd-1fcf-4370-b00b-1873622723b6" />
<img width="1335" height="669" alt="image" src="https://github.com/user-attachments/assets/a7e038d4-eb84-468b-b20d-a0b53ad700ca" />
<img width="1314" height="452" alt="image" src="https://github.com/user-attachments/assets/2d0be567-667c-4460-a732-ad2cfe302b3d" />
<img width="1237" height="642" alt="image" src="https://github.com/user-attachments/assets/6017ac34-7cab-4266-9ea0-4e70e9df05f6" />
<img width="1159" height="261" alt="image" src="https://github.com/user-attachments/assets/a5dca014-e8bd-4fb4-9f22-492496354e74" />

---
### Problems with RNN's 

<img width="1425" height="440" alt="image" src="https://github.com/user-attachments/assets/7d275e37-b379-4c8c-8a82-756724851208" />
<img width="1359" height="537" alt="image" src="https://github.com/user-attachments/assets/76093b1c-c825-4ff2-abd1-1fb0e53fd403" />
<img width="1185" height="217" alt="image" src="https://github.com/user-attachments/assets/f680fc77-8de6-402a-9918-53684e390d71" />
<img width="1294" height="658" alt="image" src="https://github.com/user-attachments/assets/c732c114-9428-47ca-8d73-963274b391dc" />

<img width="1273" height="178" alt="image" src="https://github.com/user-attachments/assets/84d45c6b-f85c-4ea1-bdd9-5e69e8ba188b" />
<img width="1363" height="666" alt="image" src="https://github.com/user-attachments/assets/5c6614ce-945f-4b22-bb63-4e1108d8ef77" />
<img width="1380" height="666" alt="image" src="https://github.com/user-attachments/assets/0d9d2526-e105-4162-b9b2-2af677b3a344" />
<img width="1401" height="301" alt="image" src="https://github.com/user-attachments/assets/eb3ff0cb-8488-4a4c-aea3-eedf5e2c9a65" />


---

## Variants of Recurrent Neural Networks (RNNs)

1. Vanilla RNN
2. Bidirectional RNNs
3. Long Short-Term Memory Networks (LSTMs)
4. Gated Recurrent Units (GRUs)

---

## 3. The Major Flaw: Vanishing Gradients

Standard RNNs struggle with "Long-Term Dependencies." If a sentence is too long, the network "forgets" the beginning because the gradient (used for training) shrinks exponentially as it backpropagates through time.

> **Analogy:** It’s like reading a book but forgetting the protagonist's name by chapter 5.

---

## 4. The Evolution: LSTM and GRU

To fix the memory problem, researchers created specialized units:

### **LSTM (Long Short-Term Memory)**

LSTMs use "Gates" to decide what to keep and what to throw away.

* **Forget Gate:** Drops irrelevant info.
* **Input Gate:** Adds new info to the cell state.
* **Output Gate:** Decides what the next hidden state should be.

### **GRU (Gated Recurrent Unit)**

A simplified, faster version of LSTM. It merges the forget and input gates into a single "update gate."

---

## 5. RNN vs. Transformers

While RNNs were the kings of NLP, they are being replaced by **Transformers** (like the ones powering me!).

* **RNN Limitation:** They process word-by-word (slow, cannot be parallelized).
* **Transformer Advantage:** They use **Attention** to look at the whole sentence at once (fast, better long-term memory).

---

## 6. Training Process (BPTT)

RNNs are trained using **Backpropagation Through Time (BPTT)**. The network is "unrolled" for all time steps, and the error is calculated and summed across every step to update the weights.

---

### Summary Table

| Feature | Standard RNN | LSTM / GRU |
| --- | --- | --- |
| **Memory** | Short-term | Long-term |
| **Complexity** | Low | High |
| **Vanishing Gradient** | High Risk | Mitigated |
| **Best Use** | Simple sequences | Complex NLP / Speech |

---

## 🧠 Human Brain Structure Sync with RNN

The human brain inspires many neural network designs. Here's how its parts relate to machine learning:

| Brain Region     | Biological Role                     | AI Analogy                          |
|------------------|--------------------------------------|-------------------------------------|
| **Brainstem**     | Controls basic life functions        | System I/O, control signals         |
| **Cerebrum**      | Higher cognitive processing          | Neural network layers               |
| └ Frontal Lobe    | Decision-making, short-term memory   | **RNN: sequence memory**            |
| └ Parietal Lobe   | Sensory integration                  | Input feature mapping               |
| └ Temporal Lobe   | Auditory, memory                     | **Weights & ANN learning**          |
| └ Occipital Lobe  | Visual processing                    | Image recognition layers            |
| **Cerebellum**    | Coordination, balance                | Fine-tuning model performance       |

🧠 [Explore brain anatomy](https://my.clevelandclinic.org/health/body/22638-brain) for deeper biological context.

---

