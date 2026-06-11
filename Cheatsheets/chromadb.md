# Chroma Cheatsheet

> ChromaDB — the open-source AI-native vector database.

---

## Installation

```bash
pip install chromadb
```

---

## Quick Start

```python
import chromadb

# In-memory (ephemeral)
client = chromadb.Client()

# Persistent (saves to disk)
client = chromadb.PersistentClient(path="./chroma_db")

# HTTP client (connect to running server)
client = chromadb.HttpClient(host="localhost", port=8000)
```

---

## Collections

```python
# Create
collection = client.create_collection(name="my_collection")

# Get existing
collection = client.get_collection(name="my_collection")

# Get or create
collection = client.get_or_create_collection(name="my_collection")

# List all
client.list_collections()

# Delete
client.delete_collection(name="my_collection")

# Count documents
collection.count()
```

### Collection with custom settings

```python
collection = client.create_collection(
    name="my_collection",
    metadata={"hnsw:space": "cosine"},   # distance: "l2" (default) | "cosine" | "ip"
    embedding_function=my_ef,            # optional custom embedding function
)
```

---

## Embedding Functions

```python
from chromadb.utils import embedding_functions

# Default (all-MiniLM-L6-v2, runs locally)
ef = embedding_functions.DefaultEmbeddingFunction()

# OpenAI
ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="sk-...",
    model_name="text-embedding-3-small",
)

# Sentence Transformers
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# HuggingFace
ef = embedding_functions.HuggingFaceEmbeddingFunction(
    api_key="hf_...",
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

# Cohere
ef = embedding_functions.CohereEmbeddingFunction(
    api_key="...",
    model_name="embed-english-v3.0",
)

# Google Generative AI
ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
    api_key="...",
)
```

---

## Add Documents

```python
collection.add(
    ids=["id1", "id2", "id3"],           # required, must be unique strings
    documents=["text one", "text two", "text three"],   # optional if embeddings provided
    embeddings=[[1.1, 2.3, ...], ...],   # optional, auto-generated if omitted
    metadatas=[{"source": "web"}, {"source": "pdf"}, {"source": "web"}],  # optional
)
```

---

## Query

```python
# Query by text (auto-embeds)
results = collection.query(
    query_texts=["search query"],
    n_results=5,
    where={"source": "web"},                         # metadata filter
    where_document={"$contains": "keyword"},         # document content filter
    include=["documents", "metadatas", "distances"], # what to return
)

# Query by embedding vector
results = collection.query(
    query_embeddings=[[1.1, 2.3, ...]],
    n_results=5,
)

# Results shape
results["ids"]         # list of lists of ids
results["documents"]   # list of lists of documents
results["metadatas"]   # list of lists of metadata dicts
results["distances"]   # list of lists of float distances
results["embeddings"]  # list of lists of embeddings (if included)
```

---

## Get Documents

```python
# Get by IDs
collection.get(ids=["id1", "id2"])

# Get with filters
collection.get(
    where={"source": "pdf"},
    where_document={"$contains": "chroma"},
    limit=10,
    offset=0,
    include=["documents", "metadatas"],
)

# Get all
collection.get()
```

---

## Update & Upsert

```python
# Update (must already exist)
collection.update(
    ids=["id1"],
    documents=["updated text"],
    metadatas=[{"source": "updated"}],
)

# Upsert (insert or update)
collection.upsert(
    ids=["id1", "id_new"],
    documents=["updated text", "brand new doc"],
    metadatas=[{"source": "a"}, {"source": "b"}],
)
```

---

## Delete

```python
# Delete by IDs
collection.delete(ids=["id1", "id2"])

# Delete by filter
collection.delete(where={"source": "web"})

# Delete by document content
collection.delete(where_document={"$contains": "keyword"})
```

---

## Metadata Filters (`where`)

```python
# Equality
{"key": "value"}
{"key": {"$eq": "value"}}

# Inequality
{"key": {"$ne": "value"}}

# Numeric comparisons
{"price": {"$gt": 10}}
{"price": {"$gte": 10}}
{"price": {"$lt": 100}}
{"price": {"$lte": 100}}

# In / not in
{"category": {"$in": ["a", "b", "c"]}}
{"category": {"$nin": ["x", "y"]}}

# Logical
{"$and": [{"source": "web"}, {"year": {"$gte": 2023}}]}
{"$or": [{"source": "web"}, {"source": "pdf"}]}
```

---

## Document Filters (`where_document`)

```python
{"$contains": "search term"}
{"$not_contains": "exclude term"}
```

---

## Running a Chroma Server

```bash
# Start server (default port 8000)
chroma run --path ./chroma_db

# With custom host/port
chroma run --path ./chroma_db --host 0.0.0.0 --port 8888
```

```python
# Connect from client
client = chromadb.HttpClient(host="localhost", port=8000)
```

---

## Distance Metrics

| Metric | `hnsw:space` value | Best for |
|--------|-------------------|----------|
| Squared L2 (default) | `"l2"` | General embeddings |
| Cosine similarity | `"cosine"` | Normalized text embeddings |
| Inner product | `"ip"` | Dot-product similarity |

> Lower distance = more similar (for `l2` and `cosine`).

---

## HNSW Index Parameters

```python
collection = client.create_collection(
    name="my_collection",
    metadata={
        "hnsw:space": "cosine",
        "hnsw:construction_ef": 100,   # build quality (default 100)
        "hnsw:search_ef": 100,         # query quality (default 10)
        "hnsw:M": 16,                  # connections per node (default 16)
        "hnsw:num_threads": 4,         # parallel threads
        "hnsw:resize_factor": 1.2,     # growth factor
    }
)
```

---

## Peek & Inspect

```python
# Preview first N items
collection.peek(limit=5)

# Count documents
collection.count()

# Collection metadata
collection.metadata
collection.name
```

---

## LangChain Integration

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# Create / load
vectorstore = Chroma(
    collection_name="my_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

# Add documents
vectorstore.add_documents(docs)

# Similarity search
results = vectorstore.similarity_search("query", k=4)
results_with_scores = vectorstore.similarity_search_with_score("query", k=4)

# As retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
```

---

## LlamaIndex Integration

```python
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, VectorStoreIndex

chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("my_collection")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
query_engine = index.as_query_engine()
```

---

## Tips & Gotchas

- IDs must be **unique strings** — duplicates are silently ignored on `add`.
- Use `upsert` instead of `add` when re-indexing to avoid duplicate errors.
- `documents` and `embeddings` are both optional, but at least one must be present.
- Default embedding model requires an internet connection on first use (downloads ~90 MB).
- For production, prefer `PersistentClient` or `HttpClient` over ephemeral `Client`.
- `n_results` cannot exceed the total number of items in the collection.
- Metadata values must be `str`, `int`, `float`, or `bool` — not nested dicts or lists.
