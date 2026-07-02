"""
PageRank assumes that the rank of a webpage W depends on the importance of a webpage 
suggested by other web pages in terms of links to the page 
i.e if a webpage 'X' has a link to webpage 'W', 'X' contributes to the importance of 'W'.

PageRank(W) = PageRank(X)/5 + PageRank(Y)/4 + PageRank(Z)/3

What would happen if no page has a link to Page 'W'?
Will its PageRank be 0?

We can add a constant (damping factor) to resolve this.

PageRank(W) = (1-d) + d * (PageRank(X)/5 + PageRank(Y)/4 + PageRank(Z)/3)

Here, we apply the same idea to SENTENCES instead of webpages — this is the
basis of the "TextRank" algorithm for extractive summarization. Instead of
hyperlinks between pages, we use COSINE SIMILARITY between sentence embeddings
as the "edges" of the graph, and let PageRank tell us which sentences are the
most "central" / important in the text.
"""

import numpy as np
import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from scipy import spatial
import networkx as nx

text = """Santiago is a Shepherd who has a recurring dream which is supposedly prophetic. Inspired on learning this, he undertakes a journey to Egypt to discover the meaning of life and fulfill his destiny. 
During the course of his travels, he learns of his true purpose and meets many characters, including an "Alchemist", that teach him valuable lessons about achieving his dreams. Santiago sets his sights on 
obtaining a certain kind of "treasure" for which he travels to Egypt. The key message is, "when you want something, all the universe conspires in helping you to achieve it." Towards the final arc, 
Santiago gets robbed by bandits who end up revealing that the "treasure" he was looking for is buried in the place where his journey began. The end."""

# --- STEP 1: Split the paragraph into sentences ---
# sent_tokenize is smarter than a naive split on '.' — it knows not to break
# on abbreviations, decimal numbers, etc.
sentences = sent_tokenize(text)

# --- STEP 2: Clean each sentence ---
# Strip punctuation/special characters so words like "Egypt." and "Egypt,"
# aren't treated as two different tokens by Word2Vec.
sentences_clean = [re.sub(r'[^\w\s]', '', s) for s in sentences]

# --- STEP 3: Remove stopwords ---
# Stopwords (the, a, is, who, etc.) carry little topical meaning and would
# just add noise to the sentence embeddings. Using a set() here instead of
# a list makes the `not in` check O(1) instead of O(n).
stop_words = set(stopwords.words('english'))
sentence_tokens = [
    [w for w in s.split() if w.lower() not in stop_words]
    for s in sentences_clean
]

# --- STEP 4: Train Word2Vec on our (tiny) corpus ---
# NOTE: gensim 4.x renamed some constructor args from the older 3.x API:
#   size -> vector_size
#   iter -> epochs
# With only 5 short sentences this is a *toy* embedding space — real
# projects would use pretrained vectors (GloVe, fastText, etc.) instead
# of training Word2Vec from scratch on so little data.
VECTOR_SIZE = 10
w2v = Word2Vec(
    sentence_tokens,
    vector_size=VECTOR_SIZE,   # dimensionality of each word vector
    min_count=1,               # keep even words that appear once
    epochs=1000                # more passes over the (very small) corpus
)

# --- STEP 5: Build one fixed-size vector per SENTENCE ---
# Important fix: the original code stored one scalar PER WORD per sentence,
# which produces vectors of different lengths (a 3-word sentence -> length 3,
# a 6-word sentence -> length 6). scipy's cosine distance requires both
# input vectors to be the SAME length, so that version crashes as soon as
# two sentences have different word counts.
#
# The standard fix: average the word vectors in a sentence together to get
# ONE fixed-size (VECTOR_SIZE-dim) vector representing that whole sentence.
# Also note: gensim 4.x moved word vectors from w2v[word] to w2v.wv[word].
sentence_embeddings = []
for words in sentence_tokens:
    if not words:
        # Guard against a sentence that becomes empty after stopword removal
        sentence_embeddings.append(np.zeros(VECTOR_SIZE))
    else:
        vecs = [w2v.wv[word] for word in words]
        sentence_embeddings.append(np.mean(vecs, axis=0))

# --- STEP 6: Build the sentence-similarity matrix ---
# similarity_matrix[i][j] = how similar sentence i is to sentence j,
# using cosine similarity (1 - cosine distance). This becomes the
# adjacency matrix of our graph — analogous to "which pages link to which".
similarity_matrix = np.zeros((len(sentence_embeddings), len(sentence_embeddings)))
for i, row in enumerate(sentence_embeddings):
    for j, col in enumerate(sentence_embeddings):
        similarity_matrix[i][j] = 1 - spatial.distance.cosine(row, col)

# --- STEP 7: Turn the similarity matrix into a graph and run PageRank ---
# Each sentence is a "node"; edge weight = similarity between two sentences.
# PageRank then scores each sentence by how "central" it is — i.e. how
# similar it is to many other important sentences in the text.
nx_graph = nx.from_numpy_array(similarity_matrix)
scores = nx.pagerank(nx_graph)   # {node_index: pagerank_score}

# --- STEP 8: Rank sentences by score, descending ---
ranked = sorted(
    ((scores[i], s) for i, s in enumerate(sentences)),
    reverse=True
)

# --- STEP 9: Pick the top 2 sentences, but print them in ORIGINAL order ---
# (Sorting by score alone would scramble the reading order of the summary.)
top_sentences = {s for _, s in ranked[:2]}
for sent in sentences:
    if sent in top_sentences:
        print(sent)



# Summary:

"""
Santiago is a Shepherd who has a recurring dream which is supposedly prophetic.
Santiago sets his sights on 
obtaining a certain kind of "treasure" for which he travels to Egypt.
"""
