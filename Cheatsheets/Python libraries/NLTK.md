**NLTK (Natural Language Toolkit)** is a powerful Python library for **text processing and NLP tasks**. Below is a structured **cheat sheet** with the most common commands and workflows you’ll use.  

---

## 📌 NLTK Cheat Sheet

### 🔹 Installation
```bash
pip install nltk
```
```python
import nltk
nltk.download('all')   # download corpora, models, stopwords, etc.
```

---

### 🔹 Tokenization
```python
from nltk.tokenize import word_tokenize, sent_tokenize

text = "NLTK makes natural language processing easy!"
words = word_tokenize(text)   # ['NLTK', 'makes', 'natural', 'language', 'processing', 'easy', '!']
sentences = sent_tokenize(text)  # ['NLTK makes natural language processing easy!']
```

---

### 🔹 Stopwords
```python
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
filtered = [w for w in words if w.lower() not in stop_words]
```

---

### 🔹 Stemming & Lemmatization
```python
from nltk.stem import PorterStemmer, WordNetLemmatizer

stemmer = PorterStemmer()
print(stemmer.stem("running"))   # run

lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("running", pos="v"))   # run
```

---

### 🔹 POS Tagging
```python
from nltk import pos_tag
tags = pos_tag(words)
# [('NLTK', 'NNP'), ('makes', 'VBZ'), ('natural', 'JJ'), ...]
```

---

### 🔹 Named Entity Recognition
```python
from nltk import ne_chunk
tree = ne_chunk(tags)
print(tree)   # hierarchical tree of named entities
```

---

### 🔹 Frequency Distribution
```python
from nltk import FreqDist
fdist = FreqDist(words)
fdist.most_common(5)   # top 5 frequent words
fdist.plot(10)         # plot top 10
```

---

### 🔹 Concordance (Keyword in Context)
```python
from nltk.book import text1
text1.concordance("sea")   # show occurrences of "sea" in Moby Dick
```

---

### 🔹 Corpora Access
```python
from nltk.corpus import brown
brown.categories()          # list categories
brown.words(categories='news')[:10]   # first 10 words in news category
```

---

### 🔹 Parsing & Chunking
```python
grammar = "NP: {<DT>?<JJ>*<NN>}"   # noun phrase
cp = nltk.RegexpParser(grammar)
result = cp.parse(tags)
print(result)
```

---

## ⚡ Key Notes
- **Tokenization** → split text into words/sentences.  
- **Stopwords** → remove common filler words.  
- **Stemming/Lemmatization** → normalize words.  
- **POS Tagging** → identify parts of speech.  
- **NER** → extract named entities.  
- **Corpora** → access built-in datasets like Brown, WordNet.  

---

## 📚 Sources
- [UMass NLTK Cheatsheet PDF](https://people.umass.edu/~sharris/in/handouts/Text-Analysis-with-NLTK-Cheatsheet.pdf)  
- [YourDevKit NLTK Cheat Sheet](https://yourdevkit.com/cheat-sheet/nltk)  
- [Cheatography NLTK Cheat Sheet](https://cheatography.com/murenei/cheat-sheets/natural-language-processing-with-python-and-nltk/pdf/?last=1527568535)

---
