# 📒 NLP Workflow Notes

## Step 1: Text Preprocessing

### 🔹 Tokenization
- **Definition**: Splitting text into smaller units (tokens) such as words or sentences.
- **Examples (NLTK)**:
```python
from nltk.tokenize import sent_tokenize, word_tokenize
text = "NLP is fun. It helps computers understand language."
print(sent_tokenize(text))   # ['NLP is fun.', 'It helps computers understand language.']
print(word_tokenize(text))   # ['NLP', 'is', 'fun', '.', 'It', 'helps', 'computers', 'understand', 'language', '.']
```

---

### 🔹 Lemmatization
- **Definition**: Reducing words to their base form (lemma) using vocabulary and grammar rules.
- **Example (SpaCy)**:
```python
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("The striped bats are hanging on their feet for best")
print([token.lemma_ for token in doc])
# ['the', 'striped', 'bat', 'be', 'hang', 'on', 'their', 'foot', 'for', 'good']
```

---

### 🔹 Stemming
- **Definition**: Cutting words to their root form (may not be linguistically correct).
- **Examples (NLTK)**:
```python
from nltk.stem import PorterStemmer, SnowballStemmer, RegexpStemmer

ps = PorterStemmer()
print(ps.stem("running"))   # run

ss = SnowballStemmer("english")  # Porter2, improved version of Porter
print(ss.stem("running"))   # run

rs = RegexpStemmer('ing$')
print(rs.stem("running"))   # runn
```

- **Comparison with Lemmatization**:
  - Stemming is **faster** (just chops suffixes/prefixes).
  - Lemmatization uses a **dictionary** → more accurate.
  - Example:  
    - Word: *history* → Stemming: *histori*, Lemmatization: *history* (correct).

- **Example 1: "sportingly"**
  - Porter: *sportingli*  
  - Snowball: *sport*  
  - ✅ Snowball is more accurate.

#### Differences:
- **Snowball Stemmer**:
  - Improved algorithm (Porter2)
  - Multilingual support
  - More accurate, balanced approach
- **Porter Stemmer**:
  - Original algorithm
  - English-focused
  - Less accurate, more nuanced

---

### 🔹 Stop-Words
- **Definition**: Common words (like *is, the, and*) removed to reduce noise.
- **Example (NLTK)**:
```python
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
words = ["This", "is", "an", "example"]
filtered = [w for w in words if w.lower() not in stop_words]
print(filtered)   # ['example']
```

## Parts of Speech: POS tagging is a fundamental task in Natural Language Processing (NLP) that involves assigning a grammatical category (such as noun, verb, adjective, etc.) to each word in a sentence. The goal is to understand the syntactic structure of a sentence and identify the grammatical roles of individual words. POS tagging provides essential information for various NLP applications, including text analysis, machine translation, and information retrieval.


POS tags are short codes representing specific parts of speech. Common POS tags include:

Noun (NN)
Verb (VB)
Adjective (JJ)
Adverb (RB)
Pronoun (PRP)
Preposition (IN)
Conjunction (CC)
Determiner (DT)
Interjection (UH)

```
import nltk
from nltk import word_tokenize, pos_tag

# Sample sentence
sentence = “The quick brown fox jumps over the lazy dog.”

# Tokenize the sentence
tokens = word_tokenize(sentence)

# Perform POS tagging
pos_tags = pos_tag(tokens)

# Display the POS tags
print(“POS Tags:”)
print(pos_tags)

Output:

POS Tags:
[(‘The’, ‘DT’), (‘quick’, ‘JJ’), (‘brown’, ‘NN’), (‘fox’, ‘NN’), (‘jumps’, ‘VBZ’), (‘over’, ‘IN’), (‘the’, ‘DT’), (‘lazy’, ‘JJ’), (‘dog’, ‘NN’), (‘.’, ‘.’)]

```

---

## Step 2: Feature Extraction

### 🔹 Bag of Words (BoW)
- **Definition**: Represents text as word frequency counts.
- **Example**:
```python
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(["NLP is fun", "NLP helps computers"])
print(X.toarray())
# [[1,1,1,0], [1,0,1,1]]
```

---

### 🔹 TF-IDF (Term Frequency – Inverse Document Frequency)
- **Definition**: Weighs words based on importance across documents.
- **Example**:
```python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(["NLP is fun", "NLP helps computers"])
print(X.toarray())
```

---

### 🔹 N-grams (Unigrams, Bigrams)
- **Definition**: Sequence of *n* words.
- **Example**:
```python
cv = CountVectorizer(ngram_range=(1,2))
X = cv.fit_transform(["NLP is fun"])
print(cv.get_feature_names_out())
# ['nlp', 'is', 'fun', 'nlp is', 'is fun']
```

---

## Step 3: Word Representations

### 🔹 Word2Vec
- **Definition**: Neural embeddings capturing semantic meaning of words.
- **Example**: *king - man + woman ≈ queen*

### 🔹 Average Word2Vec
- **Definition**: Average of word vectors to represent a sentence/document.

---

## 🔹 Deep Learning Models

- **RNN (Recurrent Neural Network)** → Handles sequential data, remembers past states.  
- **LSTM (Long Short-Term Memory)** → Solves vanishing gradient problem, remembers long dependencies.  
- **GRU (Gated Recurrent Unit)** → Simplified LSTM, faster training.  

---

## 🔹 Word Embeddings
- Dense vector representation of words (Word2Vec, GloVe, FastText).

---

## 🔹 Transformers
- **Definition**: Attention-based models for sequence processing.
- **Example**: **BERT** (Bidirectional Encoder Representations from Transformers).

---

## 🔹 Libraries
- **NLTK** → Tokenization, stemming, stop-words.  
- **SpaCy** → Advanced NLP (lemmatization, POS tagging).  
- **TensorFlow / PyTorch** → Deep learning frameworks for RNNs, LSTMs, Transformers.  

---

## Tokenization in NLP

1. **Corpus** → Collection of documents  
2. **Documents** → Sentences  
3. **Vocabulary** → Unique words  
4. **Words** → Tokens  

### NLTK Tokenizers
- **sent_tokenize** → Sentence splitting  
- **word_tokenize** → Word splitting  
- **wordpunct_tokenize** → Splits by punctuation  
- **TreebankWordTokenizer** → Penn Treebank rules  

---

## 🔹 Stemming Algorithms

| Stemmer            | Definition | Example |
|--------------------|------------|---------|
| **PorterStemmer**  | Oldest, rule-based | "running" → "run" |
| **RegexpStemmer**  | Uses regex rules | "running" → "runn" |
| **SnowballStemmer**| Improved Porter | "running" → "run" |
