Absolutely — here is a **complete, interview-oriented Vector Stores cheat sheet** in Markdown format, covering fundamentals → embeddings → similarity search → indexing → filtering → RAG → production → AWS options → comparisons.

# Vector Stores — AI/ML Engineer Cheat Sheet

> **Purpose:** Quick reference for Vector Databases / Vector Stores used in Semantic Search, RAG, Recommendation Systems, and AI Agents.

---

# 1. What is a Vector Store?

A **Vector Store** stores high-dimensional numerical representations called **embeddings** and allows efficient similarity search.

```text
Text
 ↓
Embedding Model
 ↓
Vector
 ↓
Vector Store
 ↓
Similarity Search
 ↓
Most Similar Results
```

Example:

```text
"What is AWS?"
        ↓
[0.12, -0.43, 0.77, 0.21, ...]
        ↓
     Vector DB
```

Instead of searching for exact words:

```text
"car"
```

vector search can find:

```text
"automobile"
"vehicle"
"motor vehicle"
```

because their meanings can be close in vector space.

---

# 2. Why Vector Stores?

Traditional keyword search:

```text
Query
 ↓
Keyword matching
 ↓
Documents
```

Vector search:

```text
Query
 ↓
Embedding
 ↓
Similarity Search
 ↓
Semantically similar documents
```

Useful for:

* RAG
* Semantic search
* Recommendation
* Question answering
* Document retrieval
* Image search
* Similarity detection
* AI agents
* Knowledge bases

---

# 3. Vector Store vs Vector Database

These terms are often used interchangeably.

### Vector Store

Usually refers to the abstraction/interface used to:

```text
Store
Retrieve
Search
Delete
```

vectors.

### Vector Database

A database specifically optimized for:

```text
Vector storage
Vector indexing
Similarity search
Metadata filtering
Scalability
```

Examples:

* Pinecone
* Weaviate
* Milvus
* Qdrant
* Chroma
* OpenSearch
* PostgreSQL + pgvector

---

# 4. Embeddings

An embedding converts an object into a numerical vector.

```text
Text
 ↓
Embedding Model
 ↓
Vector
```

Example:

```text
"I love machine learning"

        ↓

[0.21, -0.73, 0.18, 0.92, ...]
```

The vector may have:

```text
384 dimensions
768 dimensions
1024 dimensions
1536 dimensions
3072 dimensions
```

depending on the embedding model.

---

# 5. Semantic Meaning

Ideally:

```text
"dog"
```

and:

```text
"puppy"
```

have nearby vectors.

While:

```text
"dog"
```

and:

```text
"database"
```

are farther apart.

Conceptually:

```text
                 dog
                ●
              ●
           puppy

                         database
                            ●
```

---

# 6. Vector Store Record

A typical vector record contains:

```json
{
  "id": "doc_001",
  "vector": [0.12, -0.43, 0.77],
  "text": "AWS provides cloud computing services.",
  "metadata": {
    "source": "aws.pdf",
    "page": 10,
    "category": "cloud"
  }
}
```

Important components:

```text
ID
Vector
Text / Document
Metadata
```

---

# 7. Metadata

Metadata provides additional information about a vector.

Example:

```json
{
  "source": "research.pdf",
  "page": 12,
  "author": "John",
  "year": 2025,
  "category": "AI"
}
```

Metadata can be used for filtering.

Example:

```text
Find similar documents
BUT only:
category = "AI"
year >= 2025
```

---

# 8. Vector Search Pipeline

```text
User Query
    ↓
Embedding Model
    ↓
Query Vector
    ↓
Vector Index
    ↓
Similarity Calculation
    ↓
Top-K Results
    ↓
Optional Metadata Filtering
    ↓
Retrieved Documents
```

---

# 9. Similarity Metrics

Common metrics:

```text
1. Cosine Similarity
2. Euclidean Distance
3. Dot Product
```

---

# 10. Cosine Similarity

Measures the angle between two vectors.

Formula:

```text
             A · B
cos(A,B) = ---------
            ||A|| ||B||
```

Range:

```text
-1 → completely opposite
 0 → unrelated / orthogonal
+1 → highly similar
```

Example:

```text
A = [1, 0]
B = [1, 0]

cosine = 1
```

```text
A = [1, 0]
B = [0, 1]

cosine = 0
```

---

# 11. Euclidean Distance

Straight-line distance between vectors.

```text
d(A,B) = sqrt(
    Σ(Ai - Bi)²
)
```

Smaller distance:

```text
→ More similar
```

Example:

```text
A = [1, 2]
B = [1, 3]

distance = 1
```

---

# 12. Dot Product

```text
A · B = Σ AiBi
```

For:

```text
A = [1,2]
B = [3,4]
```

```text
A·B = 1×3 + 2×4
     = 11
```

Higher score generally means greater similarity when vectors are appropriately normalized/used with the chosen metric.

---

# 13. Cosine vs Dot Product vs Euclidean

| Metric      | Similarity/Distance | Common Use           |
| ----------- | ------------------- | -------------------- |
| Cosine      | Higher = similar    | Text embeddings      |
| Dot Product | Higher = similar    | Retrieval            |
| Euclidean   | Lower = similar     | Geometric similarity |

Important:

> The best metric depends on the embedding model and how its vectors are normalized.

---

# 14. Exact Search

Suppose we have:

```text
1 million vectors
```

For a query:

```text
Compare query against
EVERY vector
```

Complexity is roughly:

```text
O(N × D)
```

where:

```text
N = number of vectors
D = vector dimensions
```

Accurate but potentially expensive at scale.

---

# 15. Approximate Nearest Neighbor (ANN)

Instead of comparing every vector, ANN indexes the vectors so nearby candidates can be found much faster.

```text
1 billion vectors

        ↓

Index

        ↓

Candidate vectors

        ↓

Top-K
```

Trade-off:

```text
Speed ↑
Scalability ↑
Exact recall ↓ slightly
```

---

# 16. Important Vector Indexes

Know these:

```text
HNSW
IVF
PQ
Flat
```

---

# 17. HNSW

**Hierarchical Navigable Small World**

Very common in modern vector databases.

Conceptually:

```text
Layer 2

A -------- D
 \        /
  \      /

Layer 1

A -- B -- C -- D -- E
```

Higher layers provide shortcuts.

Search:

```text
Start
 ↓
Navigate coarse layer
 ↓
Move toward nearest region
 ↓
Search lower layer
 ↓
Top-K vectors
```

Advantages:

```text
Fast search
High recall
Good for dynamic data
Popular in production vector DBs
```

Trade-off:

```text
Memory usage can be high
```

---

# 18. HNSW Parameters

Important:

### M

Number of connections per node.

```text
M ↑
→ memory ↑
→ index size ↑
→ potentially recall ↑
```

### efConstruction

Controls index construction quality.

```text
efConstruction ↑
→ build time ↑
→ build memory ↑
→ potentially better recall
```

### efSearch

Controls search effort.

```text
efSearch ↑
→ search time ↑
→ recall ↑
```

---

# 19. IVF

**Inverted File Index**

Concept:

```text
Vectors
   ↓
Clusters
   ↓
Centroids
```

Example:

```text
Cluster 1 → ● ● ● ●
Cluster 2 → ● ● ●
Cluster 3 → ● ● ● ● ●
Cluster 4 → ● ●
```

Query:

```text
Query
 ↓
Find nearest clusters
 ↓
Search only those clusters
 ↓
Top-K
```

---

# 20. IVF Parameters

Important:

```text
nlist
nprobe
```

### nlist

Number of clusters.

```text
nlist ↑
→ smaller clusters
→ potentially faster search
→ more indexing complexity
```

### nprobe

Number of clusters searched.

```text
nprobe ↑
→ recall ↑
→ latency ↑
```

---

# 21. Product Quantization

PQ compresses vectors.

Original:

```text
[0.123, 0.783, 0.421, ...]
```

Instead of storing full precision, vectors are represented using compact codes.

Purpose:

```text
Memory ↓
Storage ↓
Search efficiency ↑
```

Trade-off:

```text
Compression ↑
→ precision/recall may ↓
```

---

# 22. Flat Index

Brute-force search.

```text
Query
 ↓
Compare with every vector
 ↓
Sort
 ↓
Top-K
```

Advantages:

```text
Very accurate
Simple
```

Disadvantages:

```text
Slow at very large scale
```

Useful for:

```text
Small datasets
Ground-truth evaluation
Benchmarking ANN indexes
```

---

# 23. Top-K Search

Suppose:

```text
Query = "What is AWS?"
```

Search:

```text
Top-K = 5
```

Returns:

```text
1. AWS definition       score 0.92
2. AWS services         score 0.89
3. AWS regions          score 0.84
4. Cloud computing      score 0.79
5. AWS pricing          score 0.76
```

Then those documents can be passed to an LLM.

---

# 24. RAG + Vector Store

Classic RAG:

```text
                OFFLINE
                   │
Documents
    ↓
Chunking
    ↓
Embeddings
    ↓
Vector Store
                   │
                   │
                ONLINE
                   │
User Query
    ↓
Query Embedding
    ↓
Vector Search
    ↓
Top-K Chunks
    ↓
Prompt + Context
    ↓
LLM
    ↓
Answer
```

---

# 25. Document Ingestion

Typical:

```text
PDF
 ↓
Text Extraction
 ↓
Cleaning
 ↓
Chunking
 ↓
Metadata
 ↓
Embedding
 ↓
Vector Store
```

Example chunk:

```text
Document:
AWS is a cloud computing platform...

Chunk:
"AWS provides compute, storage,
database, and machine learning services."
```

---

# 26. Chunking

Chunking divides large documents into smaller pieces.

Bad:

```text
Entire 100-page PDF
        ↓
One embedding
```

Better:

```text
100-page PDF
      ↓
Chunks
      ↓
Embedding per chunk
```

Example:

```text
chunk_size = 500 tokens
overlap = 50 tokens
```

---

# 27. Chunk Overlap

Without overlap:

```text
Chunk 1:
"AWS provides machine learning services
including SageMaker..."

Chunk 2:
"Bedrock provides foundation models..."
```

With overlap:

```text
Chunk 1:
"AWS provides machine learning services
including SageMaker and Bedrock..."

Chunk 2:
"SageMaker and Bedrock provide
different AI capabilities..."
```

Overlap helps preserve context across boundaries.

---

# 28. Chunking Trade-offs

### Small chunks

Advantages:

```text
Precise retrieval
Less irrelevant context
```

Disadvantages:

```text
May lose context
More vectors
```

### Large chunks

Advantages:

```text
More context
Fewer vectors
```

Disadvantages:

```text
More irrelevant information
Larger prompts
Potentially weaker retrieval precision
```

---

# 29. Metadata Filtering

Instead of:

```text
Search entire vector DB
```

use:

```text
category = "finance"
year >= 2025
language = "English"
```

Then perform vector search over the relevant subset.

Conceptually:

```text
Query
 ↓
Metadata Filter
 ↓
Vector Search
 ↓
Top-K
```

---

# 30. Hybrid Search

Combines:

```text
Keyword Search
+
Vector Search
```

Example:

```text
Query:
"Amazon S3 encryption"
```

Keyword search finds:

```text
"S3"
"encryption"
```

Vector search finds:

```text
"data protection in object storage"
```

Combined:

```text
Hybrid Search
 ↓
Better retrieval
```

---

# 31. Dense vs Sparse Retrieval

### Dense

Uses embeddings:

```text
Text
 ↓
Dense Vector
 ↓
Vector Search
```

Good for:

```text
Semantic meaning
Paraphrases
Conceptual similarity
```

### Sparse

Uses sparse representations such as:

```text
BM25
TF-IDF
```

Good for:

```text
Exact terms
Names
IDs
Rare keywords
Technical terminology
```

---

# 32. Hybrid Retrieval Architecture

```text
                 Query
                   │
          ┌────────┴────────┐
          ▼                 ▼
      BM25 Search      Vector Search
          │                 │
          └────────┬────────┘
                   ▼
              Fusion/Rerank
                   │
                   ▼
                 Top-K
```

---

# 33. Reranking

Initial retrieval:

```text
100 documents
```

Vector search:

```text
100 → Top 20
```

Reranker:

```text
20 → Top 5
```

Architecture:

```text
Query
 ↓
Vector Search
 ↓
Candidate Documents
 ↓
Reranker
 ↓
Best Documents
 ↓
LLM
```

Reranking improves retrieval quality at the cost of additional latency/compute.

---

# 34. Retrieval Metrics

Important:

### Precision

```text
Relevant retrieved
------------------
All retrieved
```

### Recall

```text
Relevant retrieved
------------------
All relevant documents
```

### Recall@K

How many relevant documents are found among the top K results.

### MRR

**Mean Reciprocal Rank**

Useful when the position of the first relevant result matters.

---

# 35. RAG Evaluation

Evaluate separately:

```text
Retrieval
   +
Generation
```

### Retrieval

Check:

```text
Recall@K
Precision@K
MRR
NDCG
```

### Generation

Check:

```text
Faithfulness
Answer relevance
Correctness
Context utilization
Hallucination
```

---

# 36. Vector Store CRUD

Most vector stores support:

```text
Create
Read/Search
Update
Delete
```

Conceptually:

```python
vector_store.add_documents(documents)

vector_store.similarity_search(
    query,
    k=5
)

vector_store.delete(ids)
```

---

# 37. Common Vector Stores

## Pinecone

Managed vector database.

Good for:

```text
Production RAG
Semantic search
Managed infrastructure
```

---

## Weaviate

Open-source + managed options.

Features:

```text
Vector search
Hybrid search
Metadata filtering
RAG
```

---

## Milvus

Open-source vector database.

Good for:

```text
Large-scale vector search
High-performance workloads
Self-hosting
```

---

## Qdrant

Vector database focused on:

```text
Similarity search
Filtering
Payload metadata
RAG
```

---

## Chroma

Developer-friendly vector database/store.

Good for:

```text
Prototypes
Local RAG
Development
Small applications
```

---

## FAISS

Facebook AI Similarity Search.

Important:

> FAISS is primarily a vector similarity search/indexing library, not a complete production database by itself.

Good for:

```text
Local search
Research
Prototyping
Custom retrieval systems
```

---

## OpenSearch

AWS-compatible search platform with vector capabilities.

Useful for:

```text
AWS RAG
Hybrid search
Vector search
Enterprise search
```

---

## PostgreSQL + pgvector

PostgreSQL extension for vector similarity search.

Useful when:

```text
You already use PostgreSQL
Need relational + vector data
Want SQL + vector search
```

---

# 38. Vector Store Comparison

| Technology | Type                 | Best For                 |
| ---------- | -------------------- | ------------------------ |
| Pinecone   | Managed              | Production RAG           |
| Weaviate   | Vector DB            | RAG/search               |
| Milvus     | Vector DB            | Large-scale workloads    |
| Qdrant     | Vector DB            | Search + filtering       |
| Chroma     | Vector store         | Prototypes               |
| FAISS      | Library              | Local/research           |
| OpenSearch | Search engine        | AWS/vector/hybrid search |
| pgvector   | PostgreSQL extension | SQL + vectors            |

---

# 39. FAISS Example

Install:

```bash
pip install faiss-cpu
```

Create index:

```python
import faiss
import numpy as np

dimension = 768

index = faiss.IndexFlatL2(dimension)

vectors = np.random.random(
    (1000, dimension)
).astype("float32")

index.add(vectors)
```

Search:

```python
query = np.random.random(
    (1, dimension)
).astype("float32")

distances, indices = index.search(
    query,
    5
)
```

Returns:

```text
Top 5 nearest vectors
```

---

# 40. LangChain + Vector Store

Typical:

```python
from langchain_chroma import Chroma

vectorstore = Chroma.from_documents(
    documents,
    embedding
)

results = vectorstore.similarity_search(
    "What is AWS?",
    k=5
)
```

General LangChain flow:

```text
Documents
 ↓
Embedding
 ↓
VectorStore
 ↓
Retriever
 ↓
LLM
```

---

# 41. Retriever

A retriever abstracts search.

```python
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)
```

Then:

```python
docs = retriever.invoke(
    "What is machine learning?"
)
```

Conceptually:

```text
Retriever
    ↓
Vector Store
    ↓
Search
    ↓
Documents
```

---

# 42. Vector Store vs Retriever

Important distinction:

```text
Vector Store
→ stores/searches vectors

Retriever
→ defines how relevant documents are retrieved
```

Example:

```text
Vector Store
    ↓
Retriever
    ↓
Top-K documents
```

A retriever may use:

```text
Vector search
Keyword search
Hybrid search
Metadata filtering
Reranking
```

---

# 43. Distance vs Similarity

Be careful.

Similarity:

```text
Higher = better
```

Distance:

```text
Lower = better
```

For example:

```text
Cosine similarity:
0.95 → excellent

Euclidean distance:
0.10 → very close
```

Don't compare raw scores from different databases/metrics directly.

---

# 44. Vector Dimensions

If embedding model outputs:

```text
768 dimensions
```

every vector inserted into that index normally needs:

```text
768 dimensions
```

This is invalid:

```text
Index = 768 dimensions

Vector = 1536 dimensions
```

You must use a compatible dimension.

---

# 45. Embedding Model Consistency

Important production rule:

```text
Indexing embedding model
        =
Query embedding model
```

Example:

```text
Documents → Model A
Queries   → Model B
```

Potentially problematic.

Better:

```text
Documents → Model A
Queries   → Model A
```

Unless the retrieval architecture specifically supports compatible asymmetric encoders.

---

# 46. Updating Embeddings

If you change embedding models:

```text
Old model
 ↓
Old vectors

New model
 ↓
New vectors
```

Usually you need to:

```text
Re-embed documents
 ↓
Create/rebuild index
 ↓
Evaluate retrieval
 ↓
Switch production
```

---

# 47. Namespace / Collection

Many vector systems organize vectors into:

```text
Collections
Namespaces
Indexes
Partitions
```

Example:

```text
knowledge-base
│
├── HR
├── Finance
├── Engineering
└── Legal
```

This helps with:

```text
Isolation
Filtering
Multi-tenancy
Organization
```

---

# 48. Multi-Tenant Vector Store

Suppose:

```text
Company A
Company B
Company C
```

Never allow:

```text
Company A query
       ↓
Company B documents
```

Use:

```text
tenant_id = company_A
```

during retrieval.

```text
Query
 ↓
tenant_id filter
 ↓
Vector Search
 ↓
Authorized documents
```

This is critical for production RAG security.

---

# 49. Vector Database Scaling

At scale:

```text
10K vectors
    ↓
100K
    ↓
1M
    ↓
100M
    ↓
1B+
```

Consider:

```text
Index type
Sharding
Replication
Memory
Storage
Latency
Throughput
Filtering
Partitioning
```

---

# 50. Latency Optimization

Ways to reduce retrieval latency:

```text
Use ANN
 ↓
Reduce candidate count
 ↓
Tune HNSW/IVF
 ↓
Metadata filtering
 ↓
Cache frequent queries
 ↓
Reduce vector dimensions when appropriate
 ↓
Use efficient infrastructure
```

---

# 51. Recall vs Latency

Classic trade-off:

```text
Recall ↑
    │
    │       ●
    │     ●
    │   ●
    │ ●
    └────────────────→
             Latency ↑
```

Increasing search effort generally:

```text
Recall ↑
Latency ↑
```

Tune according to application requirements.

---

# 52. Vector Quantization

Quantization reduces vector representation size.

Example:

```text
float32
 ↓
float16
 ↓
int8
 ↓
compressed representation
```

Benefits:

```text
Memory ↓
Storage ↓
Potentially latency ↓
```

Potential drawback:

```text
Retrieval quality ↓
```

Always benchmark.

---

# 53. Filtering + ANN

Important production concept.

Suppose:

```text
1 billion vectors
```

But query requires:

```text
country = India
department = AI
year >= 2025
```

Ideal system should efficiently combine:

```text
Metadata filtering
+
Vector search
```

Filtering behavior varies by vector database/index implementation, so benchmark the actual workload.

---

# 54. RAG Failure Modes

### 1. Bad chunking

```text
Wrong chunks
 ↓
Wrong retrieval
 ↓
Bad answer
```

### 2. Poor embeddings

```text
Poor semantic representation
 ↓
Wrong results
```

### 3. Wrong K

```text
K too small → missing context

K too large → noisy context
```

### 4. No metadata filtering

```text
Irrelevant documents
 ↓
LLM confusion
```

### 5. No reranking

```text
Relevant document may rank too low
```

---

# 55. Advanced Retrieval Pipeline

Production RAG:

```text
User Query
    ↓
Query Rewriting
    ↓
Query Embedding
    ↓
Hybrid Search
    ↓
Metadata Filtering
    ↓
Top 50
    ↓
Reranker
    ↓
Top 5
    ↓
Context Compression
    ↓
LLM
    ↓
Answer
```

---

# 56. Query Rewriting

User asks:

```text
"How does it work?"
```

Previous context:

```text
AWS SageMaker
```

Rewrite:

```text
"How does Amazon SageMaker work?"
```

Then perform retrieval.

Useful for:

```text
Conversational RAG
```

---

# 57. Multi-Query Retrieval

One query:

```text
"What is AWS?"
```

Generate:

```text
"What is Amazon Web Services?"
"What services does AWS provide?"
"How does AWS cloud computing work?"
```

Search each query:

```text
Query 1 → results
Query 2 → results
Query 3 → results
```

Combine and rerank.

---

# 58. Parent-Child Retrieval

Instead of embedding large documents directly:

```text
Parent Document
     │
     ├── Child Chunk 1
     ├── Child Chunk 2
     └── Child Chunk 3
```

Search child chunks:

```text
Query
 ↓
Child chunk
```

Then return:

```text
Parent context
```

This can improve retrieval precision while preserving larger context.

---

# 59. Contextual Retrieval

Instead of embedding:

```text
"Revenue increased by 20%."
```

add contextual information:

```text
"Acme Corporation's Q4 2025 revenue
increased by 20%."
```

Then embed.

Goal:

```text
Chunk meaning becomes more self-contained.
```

---

# 60. Metadata Best Practices

Store useful metadata:

```json
{
  "document_id": "doc123",
  "source": "annual_report.pdf",
  "page": 14,
  "section": "Revenue",
  "document_type": "financial",
  "created_at": "2025-12-01",
  "tenant_id": "company_a"
}
```

Avoid excessive metadata that isn't useful for:

```text
Filtering
Security
Citation
Debugging
```

---

# 61. Production RAG Architecture

```text
                         USER
                           │
                           ▼
                      API Gateway
                           │
                           ▼
                       Backend
                           │
                ┌──────────┴──────────┐
                │                     │
                ▼                     ▼
          Query Processing       Conversation DB
                │
                ▼
          Embedding Model
                │
                ▼
        ┌──────────────────┐
        │   Vector Store   │
        └────────┬─────────┘
                 │
                 ▼
            Top 50 chunks
                 │
                 ▼
              Reranker
                 │
                 ▼
              Top 5
                 │
                 ▼
                 LLM
                 │
                 ▼
               Answer
```

---

# 62. AWS Vector Store Architecture

A common AWS-oriented RAG design:

```text
                 Documents
                     │
                     ▼
                    S3
                     │
                     ▼
              Processing Layer
                     │
                     ▼
                Embeddings
                     │
                     ▼
              OpenSearch /
              pgvector /
              other vector DB
                     ▲
                     │
User ──→ API ──→ Retrieval
                     │
                     ▼
                  Bedrock
                     │
                     ▼
                  Answer
```

---

# 63. Security

Production vector stores should consider:

```text
Authentication
Authorization
Encryption
Network isolation
Tenant isolation
Metadata filtering
Access control
Audit logs
Secrets management
```

Never rely solely on:

```text
"the LLM will not show unauthorized documents"
```

Authorization must happen **before context reaches the LLM**.

---

# 64. Vector Store + AI Agent

Agent architecture:

```text
User
 ↓
Agent
 ↓
Reasoning
 ↓
Choose Tool
 ├── Vector Search
 ├── Database
 ├── API
 └── Calculator
 ↓
Observation
 ↓
Reasoning
 ↓
Final Answer
```

Vector Store acts as the agent's:

```text
Knowledge retrieval tool
```

---

# 65. Vector Store vs Traditional Database

| Feature           | Traditional DB   | Vector DB      |
| ----------------- | ---------------- | -------------- |
| Exact lookup      | Excellent        | Possible       |
| SQL               | Excellent        | Depends        |
| Joins             | Excellent        | Limited/varies |
| Semantic search   | Poor             | Excellent      |
| Vector similarity | No/native varies | Core feature   |
| Metadata          | Excellent        | Supported      |
| RAG               | Not ideal alone  | Excellent      |
| Transactions      | Strong           | Depends        |

Modern systems can combine both.

---

# 66. Vector DB vs Graph DB

### Vector DB

Good for:

```text
Semantic similarity
RAG
Nearest-neighbor search
```

### Graph DB

Good for:

```text
Relationships
Entities
Knowledge graphs
Multi-hop reasoning
```

Hybrid architecture:

```text
Query
 ├── Vector Search
 └── Graph Search
       ↓
   Combine Context
       ↓
      LLM
```

---

# 67. Important Interview Questions

### Q1. What is a vector database?

> A database optimized for storing embeddings and performing efficient similarity search over high-dimensional vectors.

### Q2. Why use vector search?

> It retrieves semantically similar information rather than relying only on exact keyword matching.

### Q3. What is an embedding?

> A numerical vector representation of data that captures semantic or learned relationships.

### Q4. What is HNSW?

> An approximate nearest-neighbor indexing algorithm that uses a hierarchical graph structure to efficiently search for similar vectors.

### Q5. Why ANN?

> Brute-force search becomes expensive as the number of vectors grows; ANN provides much faster search with a controllable recall trade-off.

### Q6. Cosine vs Euclidean?

> Cosine compares vector orientation, while Euclidean measures geometric distance. The appropriate choice depends on the embedding model and normalization.

### Q7. What is top-K?

> The K most relevant vectors/documents returned by a similarity search.

### Q8. What is metadata filtering?

> Restricting search results based on structured attributes such as tenant, date, category, or document type.

### Q9. What is hybrid search?

> Combining lexical/keyword retrieval with semantic vector retrieval.

### Q10. Why reranking?

> Initial retrieval efficiently produces candidates; a more expensive reranker can then select the most relevant documents.

---

# 68. Interview: Design a RAG System

Answer structure:

```text
1. Documents
      ↓
2. Object storage
      ↓
3. Text extraction
      ↓
4. Chunking
      ↓
5. Embeddings
      ↓
6. Vector DB
      ↓
7. Query embedding
      ↓
8. Similarity/hybrid search
      ↓
9. Metadata filtering
      ↓
10. Reranking
      ↓
11. Prompt construction
      ↓
12. LLM
      ↓
13. Answer + citations
```

Then discuss:

```text
Latency
Recall
Precision
Security
Multi-tenancy
Cost
Monitoring
Evaluation
```

---

# 69. Quick Decision Guide

```text
Need local prototype?
        ↓
     Chroma / FAISS

Need managed production vector DB?
        ↓
     Pinecone

Need open-source scalable DB?
        ↓
     Milvus / Qdrant / Weaviate

Already using PostgreSQL?
        ↓
     pgvector

AWS search/RAG ecosystem?
        ↓
     OpenSearch

Need pure similarity-search library?
        ↓
     FAISS
```

---

# 70. Must-Know Terms

Memorize these:

```text
Embedding
Vector
Dimension
Similarity
Distance
Cosine Similarity
Dot Product
Euclidean Distance
Nearest Neighbor
ANN
HNSW
IVF
PQ
Top-K
Metadata
Filtering
Dense Retrieval
Sparse Retrieval
BM25
Hybrid Search
Reranking
Recall@K
Precision@K
MRR
NDCG
Chunking
Chunk Overlap
Query Rewriting
Multi-Query Retrieval
Parent-Child Retrieval
Contextual Retrieval
Vector Quantization
Sharding
Replication
Index
Namespace
Collection
Multi-tenancy
```

---

# 71. One-Minute Revision

```text
Vector Store
    ↓
Stores embeddings
    ↓
Embedding Model
    ↓
Text → Vector
    ↓
Similarity Search
    ↓
Cosine / Dot / Euclidean
    ↓
ANN
    ↓
HNSW / IVF / PQ
    ↓
Top-K
    ↓
Metadata Filtering
    ↓
Hybrid Search
    ↓
Reranking
    ↓
RAG
    ↓
LLM
```

---

# 72. Golden Mental Model

Remember:

```text
             DOCUMENTS
                 │
                 ▼
             CHUNKING
                 │
                 ▼
            EMBEDDINGS
                 │
                 ▼
          ┌──────────────┐
          │ VECTOR STORE │
          └──────┬───────┘
                 │
                 │
              SEARCH
                 ▲
                 │
             USER QUERY
                 │
                 ▼
            EMBEDDING
                 │
                 ▼
           SIMILARITY
                 │
                 ▼
              TOP-K
                 │
                 ▼
             RERANK
                 │
                 ▼
               LLM
                 │
                 ▼
              ANSWER
```

> **Core idea:**
> **Embedding converts meaning → vectors. Vector stores make those vectors searchable. Retrieval finds relevant context. RAG gives that context to an LLM.**
