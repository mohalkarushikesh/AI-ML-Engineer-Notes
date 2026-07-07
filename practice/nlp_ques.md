Here are hands-on NLP practice exercises organized by level. Each one is a mini-project you can actually build and run.

## Beginner

1. **Text cleaning pipeline** — Take a paragraph and write code to lowercase it, remove punctuation, strip stopwords, and tokenize into words.
2. **Word frequency counter** — Read a `.txt` file (e.g., a book from Project Gutenberg) and output the 20 most common words, then the 20 most common after removing stopwords.
3. **Stemming vs. lemmatization** — Run the same sentence through a stemmer (Porter) and a lemmatizer (WordNet) and compare the outputs. Explain the differences you see.
4. **Bag-of-Words by hand** — Build a document-term matrix for 3–4 short sentences without using a library, then verify with `CountVectorizer`.
5. **Simple sentiment classifier** — Use a small labeled dataset (e.g., movie reviews) with `CountVectorizer` + Logistic Regression / Naive Bayes and measure accuracy.
6. **N-gram generator** — Write a function that produces all bigrams and trigrams from a sentence.
7. **Regex extraction** — Extract all emails, phone numbers, and hashtags from a block of messy text.

## Medium

1. **TF-IDF search engine** — Build a tiny search engine over a set of documents: rank documents by TF-IDF cosine similarity to a query.
2. **Named Entity Recognition** — Use spaCy to extract people, organizations, and locations from news articles; then visualize the entity distribution.
3. **POS-tag–based analysis** — Tag a corpus and answer questions like "which adjectives most often precede a given noun?"
4. **Text classification with word embeddings** — Represent documents by averaging pre-trained Word2Vec/GloVe vectors, then train a classifier and compare against TF-IDF.
5. **Topic modeling** — Apply LDA to a document collection and interpret the top words per topic.
6. **Spelling correction** — Implement a Norvig-style spell corrector using edit distance and word frequencies.
7. **Language detection** — Build a classifier that identifies the language of a sentence using character n-grams.
8. **Text summarization (extractive)** — Score sentences by importance (TF-IDF or TextRank) and return the top few as a summary.

## Advanced

1. **Fine-tune a transformer** — Fine-tune BERT/DistilBERT on a custom classification task (e.g., intent detection) using Hugging Face; report F1 and a confusion matrix.
2. **Build a Q&A system** — Implement extractive question answering: given a passage and a question, return the answer span using a pre-trained model.
3. **Sequence-to-sequence model** — Train a small transformer or LSTM for a task like translation, date normalization, or abstractive summarization.
4. **NER from scratch (custom entities)** — Annotate your own data and train a custom NER model for a domain (e.g., medical terms, product names).
5. **Retrieval-Augmented Generation (RAG)** — Chunk documents, embed them into a vector store (FAISS/Chroma), retrieve relevant chunks, and feed them to an LLM to answer questions.
6. **Semantic similarity service** — Use sentence embeddings (Sentence-BERT) to build a duplicate-question or paraphrase detector.
7. **Attention visualization** — Extract and plot attention weights from a transformer to interpret which tokens it focuses on.
8. **Evaluate & compare LLM prompts** — Design a benchmark, run multiple prompting strategies (zero-shot, few-shot, chain-of-thought), and quantify differences.

A good learning path is to pick one exercise per level and complete it end-to-end (data → code → evaluation → short write-up) before moving on. 

Want me to expand any single exercise into a full step-by-step project with starter code and a dataset suggestion?
