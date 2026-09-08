**NLP cheatsheet** 🧠📑 

---

## 🔤 Text Preprocessing
- **Tokenization** → Split text into words/subwords.
- **Stopword Removal** → Drop common words (the, is, etc.).
- **Stemming** → Reduce words to root form (e.g., *running → run*).
- **Lemmatization** → Normalize to dictionary form (better than stemming).
- **Lowercasing** → Standardize text.
- **Normalization** → Remove punctuation, special chars, accents.

---

## 📊 Feature Extraction
- **Bag of Words (BoW)** → Word counts.
- **TF‑IDF** → Weighted word importance.
- **Word Embeddings** → Dense vectors (Word2Vec, GloVe, FastText).
- **Contextual Embeddings** → BERT, GPT, Transformer-based.

---

## 🧩 Common NLP Tasks
- **Text Classification** → Spam detection, sentiment analysis.
- **Named Entity Recognition (NER)** → Extract names, dates, places.
- **Part-of-Speech Tagging (POS)** → Identify nouns, verbs, etc.
- **Machine Translation** → Translate languages.
- **Summarization** → Shorten text meaningfully.
- **Question Answering (QA)** → Extract/Generate answers.
- **Text Generation** → Chatbots, creative writing.

---

## ⚙️ Key Models
- **Classical** → Naive Bayes, Logistic Regression, SVM.
- **Deep Learning** → RNN, LSTM, GRU, CNN.
- **Transformers** → BERT, GPT, T5, RoBERTa, XLNet.

---

## 🛠️ Libraries & Tools
- **NLTK** → Preprocessing, classical NLP.
- **spaCy** → Fast, industrial NLP.
- **scikit-learn** → ML + text features.
- **gensim** → Word2Vec, topic modeling.
- **Transformers (HuggingFace)** → Pretrained models.
- **OpenAI / Microsoft Copilot APIs** → Advanced LLMs.

---

## 📐 Evaluation Metrics
- **Classification** → Accuracy, Precision, Recall, F1.
- **Translation/Summarization** → BLEU, ROUGE, METEOR.
- **Language Models** → Perplexity.

---
- BLEU = Bilingual Evaluation Understudy
- ROUGE = Recall-Oriented Understudy for Gisting Evaluation

- Precision = of the documents you retrieved, how many were relevant?
→ relevant retrieved / total retrieved
- Recall = of all the relevant documents that exist, how many did you find?
→ relevant retrieved / total relevant in the collection
 
- BLEU — precision-oriented. Of the n-grams the machine produced, how many appear in the reference? Used mainly for machine translation. Asks: "Is what I generated correct?"
- ROUGE — recall-oriented. Of the n-grams in the reference, how many did the machine capture? Used mainly for summarization. Asks: "Did I cover what I should have?"

The logic behind the split:
- Translation → you don't want to add wrong/extra words → penalize junk → precision (BLEU)
- Summarization → you don't want to miss key content → reward coverage → recall (ROUGE)
 
---

## 🚀 Quick Workflow
1. Collect text data  
2. Preprocess (clean, tokenize, normalize)  
3. Feature extraction (BoW, TF‑IDF, embeddings)  
4. Train model (classical ML or deep learning)  
5. Evaluate with metrics  
6. Deploy (API, chatbot, app)  

---

<img width="1024" height="1536" alt="BCO edf33e3d-1a38-459a-8d93-108e22095499" src="https://github.com/user-attachments/assets/b6451e1a-5058-4729-ae87-49c222b2f3bc" />
