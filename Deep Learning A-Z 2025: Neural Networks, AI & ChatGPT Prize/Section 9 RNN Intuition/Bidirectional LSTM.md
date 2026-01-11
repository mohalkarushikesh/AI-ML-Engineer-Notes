# 🔁 Bidirectional LSTM (BiLSTM)

## 📌 Definition
- A **Bidirectional LSTM** is an extension of the standard LSTM that processes input sequences in **both directions**:
  - **Forward pass** → from past to future  
  - **Backward pass** → from future to past  
- The outputs from both directions are combined (concatenated or summed) to capture **context from both past and future**.

---

## 🧬 Architecture
- Two LSTM layers run in parallel:
  - **Forward LSTM**: reads sequence from $t=1$ to $t=T$  
  - **Backward LSTM**: reads sequence from $t=T$ to $t=1$  
- Final hidden state:  
  $$h_t = [h_t^{forward}; \; h_t^{backward}]$$  
  (concatenation of forward and backward hidden states)

<img width="850" height="444" alt="Bidirectional-LSTM-architecture" src="https://github.com/user-attachments/assets/40a79416-8954-4acc-a798-c0bb24887042" />

---

## 📈 Why Use BiLSTM?
- Standard LSTM only remembers **past context**.  
- BiLSTM adds **future context**, making it powerful for tasks where meaning depends on both sides of a word or event.  
- Example: In the sentence *“He went to the bank to deposit money”*, the word *bank* is clarified by **future context** (“deposit money”).

---

## ✅ Advantages
- Captures **complete context** (past + future).  
- Improves accuracy in NLP tasks like sentiment analysis, translation, speech recognition.  
- Handles ambiguous sequences better.

---

## ⚠️ Limitations
- **Slower training** (two LSTMs instead of one).  
- **Not suitable for real-time prediction** (future context unavailable).  
- Higher memory usage.

---

## 📌 Applications
- **Natural Language Processing (NLP)** → text classification, machine translation, named entity recognition.  
- **Speech Recognition** → phoneme prediction.  
- **Bioinformatics** → DNA/protein sequence analysis.  

---

## 📝 Summary
- BiLSTM = LSTM + backward context.  
- Formula:  
  $$h_t = [h_t^{forward}; h_t^{backward}]$$  
- Best for tasks where **future context matters**.  
- Trade-off: more computation, not real-time friendly.

---
