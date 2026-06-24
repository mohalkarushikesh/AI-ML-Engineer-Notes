re-ranker methods: 

Term			Meaning
Cross-Encoder		How inputs are processed (query + doc together)
Deep Model		The type of model (neural network like BERT)


BM25 is a ranking algorithm that estimates document relevance using term frequency, term rarity (IDF), and document length normalization.

👉 Problem with older methods (TF‑IDF):

Longer documents naturally contain more words
So they often get higher scores unfairly

👉 What BM25 does:

It reduces the score for very long documents
So a short, focused document can rank higher than a long, irrelevant one


- Repeating a word many times doesn't always mean more relevance
- Long documents shouldn’t dominate unfairly
