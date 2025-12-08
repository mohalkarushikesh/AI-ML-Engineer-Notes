#### FAISS (Facebook AI Similarity Search) is an open-source library for fast similarity search and clustering of high-dimensional vectors, essential for finding similar items in large datasets, powering recommendation systems, image/text search, and NLP, offering CPU/GPU support and efficient handling of vectors that don't fit in RAM via compression and optimized indexing. Developed by Meta (Facebook), it's a crucial tool for AI/ML, providing high-performance nearest-neighbor search for massive datasets. 
This video provides an introduction to FAISS and its key indexes:

#### Key Features & Functions
- Vector Search: Quickly finds similar vectors (e.g., embeddings) in huge collections.
- Clustering: Groups similar vectors together.
- Scalability: Handles datasets larger than RAM.
- Efficiency: Uses compression (vector codecs) and optimized indexes (like IVF) for speed.
- GPU Support: Offers fast implementations for Nvidia GPUs.
- Accessibility: Available via Python (NumPy) and C++, with easy installation via Conda/pip. 

#### Common Use Cases
- Recommendation Systems: Finding similar users or items.
- Image/Video Retrieval: Searching for visually similar content.
= Document Search: Finding similar articles or texts (semantic search).
- Natural Language Processing (NLP): Searching for related words or sentences. 

#### How it Works (Simplified)
- Embeddings: Your data (text, images) is converted into dense numerical vectors (embeddings).
- Indexing: FAISS builds specialized data structures (indexes) from these vectors, often compressing them.
- Querying: A query vector is used to search the index for the most similar vectors (nearest neighbors). 

#### Getting Started
- Install: conda install -c pytorch faiss-gpu (for GPU) or faiss-cpu (for CPU).
- Learn More: Visit the official documentation for tutorials and examples. 

---

# 🚀 Getting Started with FAISS (Facebook AI Similarity Search)

FAISS helps you perform efficient similarity search and clustering of dense vectors.

---

## 1. Installation

**CPU-only version:**
```bash
pip install faiss-cpu
```

**GPU-accelerated version (requires CUDA):**
```bash
pip install faiss-gpu
```

**Verify installation:**
```python
import faiss
print(faiss.__version__)
```

---

## 2. Data Preparation

- Represent your items (images, text, etc.) as **numerical vectors (embeddings)** using pre-trained models (e.g., BERT for text, CNNs for images).  
- Ensure vectors are in **float32 format** and stored as **NumPy arrays**.

---

## 3. Building a FAISS Index

### Choose an index type
- **`IndexFlatL2`** → exact nearest neighbor search (good for smaller datasets).  
- **`IndexIVFFlat`** → approximate nearest neighbor search (scalable for larger datasets, requires training).  

### Initialize the index
```python
dimension = 128  # Example dimension of your vectors
index = faiss.IndexFlatL2(dimension)
```

### Add vectors to the index
```python
index.add(your_vectors_array)
```

### Example with `IndexIVFFlat` (requires training)
```python
nlist = 100  # Number of centroids
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)

# Train before adding data
index.train(training_vectors_array)
index.add(your_vectors_array)
```

---

## 4. Performing Similarity Search

### Prepare your query vector(s)
```python
query_vector = your_query_vector_array
```

### Perform the search
```python
k = 5  # Number of nearest neighbors to retrieve
distances, indices = index.search(query_vector, k)
```

- `distances` → distances to the nearest neighbors  
- `indices` → original indices of the nearest neighbors in your dataset  

---

## 🔧 Example with `IndexFlatL2`

```python
import faiss
import numpy as np

# 1. Generate some dummy data (e.g., 100 vectors of 64 dimensions)
dimension = 64
num_vectors = 100
vectors = np.random.rand(num_vectors, dimension).astype('float32')

# 2. Create a FlatL2 index
index = faiss.IndexFlatL2(dimension)

# 3. Add vectors to the index
index.add(vectors)

# 4. Create a query vector
query_vector = np.random.rand(1, dimension).astype('float32')

# 5. Perform a search for the 5 nearest neighbors
k = 5
distances, indices = index.search(query_vector, k)

print("Distances to nearest neighbors:", distances)
print("Indices of nearest neighbors:", indices)
```

---
