## BERT (Bidirectional Encoder Representations from Transformers) — Notes

### 1. What is BERT?

**BERT** is a **pre-trained Transformer encoder model** developed by **Google (2018)**.
It learns **bidirectional context** (left + right) simultaneously, unlike GPT (left-to-right).

👉 Best for **understanding tasks** (NLU), not text generation.

---

### 2. Key Idea (Why BERT is powerful)

Traditional models:

* Read text **left → right** or **right → left**

BERT:

* Reads **both directions at once**
* Understands **true context**

Example:

> *“He went to the bank to deposit money”*
> *“He sat on the river bank”*

BERT understands **bank** differently based on context.

---

### 3. Architecture

* **Transformer Encoder only**
* Multi-head self-attention
* Deep bidirectional representations

| BERT Variant | Layers | Hidden Size | Heads |
| ------------ | ------ | ----------- | ----- |
| BERT-base    | 12     | 768         | 12    |
| BERT-large   | 24     | 1024        | 16    |


<img width="640" height="509" alt="image" src="https://github.com/user-attachments/assets/079765f5-e23d-4879-b2f2-8a52557ce0fb" />

---

### 4. Input Representation

BERT input = **Token + Segment + Position embeddings**

Example:

```
[CLS] I love AI [SEP] BERT is powerful [SEP]
```

* **[CLS]** → Classification token
* **[SEP]** → Sentence separator
* **Segment Embeddings** → Sentence A / B
* **Position Embeddings** → Token order

---

### 5. Pre-training Tasks

#### 1️⃣ Masked Language Model (MLM)

* Randomly mask **15%** of tokens
* Predict masked words

Example:

```
Input: I love [MASK]
Output: AI
```

> This enables **bidirectional learning**

---

#### 2️⃣ Next Sentence Prediction (NSP)

* Predict whether sentence B follows sentence A

Example:

```
A: I love AI.
B: It is very powerful.  → IsNext
```

---

### 6. Fine-tuning

Same BERT model, small task-specific head added.

| Task                     | Output           |
| ------------------------ | ---------------- |
| Text classification      | [CLS] token      |
| Named Entity Recognition | Token-wise       |
| Question Answering       | Start & end span |
| Sentence similarity      | CLS embedding    |

---

### 7. Where BERT is Used

✅ Sentiment analysis
✅ Question answering (SQuAD)
✅ NER
✅ Semantic search
✅ Text classification
❌ Text generation (use GPT instead)

---

### 8. Popular BERT Variants

* **RoBERTa** → No NSP, more data
* **DistilBERT** → Faster, smaller
* **ALBERT** → Parameter sharing
* **SciBERT** → Scientific text
* **BioBERT** → Biomedical domain
* **mBERT** → Multilingual

---

### 9. Advantages

✔ Deep bidirectional context
✔ Strong transfer learning
✔ High accuracy on NLP benchmarks

---

### 10. Limitations

❌ Slow inference
❌ Heavy model
❌ Not good for generation
❌ Context length limited (512 tokens)

---

### 11. BERT vs GPT (Quick)

| Feature   | BERT          | GPT           |
| --------- | ------------- | ------------- |
| Direction | Bidirectional | Left-to-right |
| Type      | Encoder       | Decoder       |
| Best for  | Understanding | Generation    |

---

### 12. Simple Intuition

> **BERT = “Read the whole sentence before answering”**

---

If you want:

* 🔥 **Interview-ready BERT explanations**
* 🧠 **Math behind self-attention**
* 🛠 **BERT fine-tuning code (PyTorch/HF)**
* 📄 **One-page PDF notes**
