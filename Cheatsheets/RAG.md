# RAG (Retrieval-Augmented Generation) Cheatsheet

---

## What is RAG?

RAG enhances LLM responses by **retrieving relevant external documents** at inference time and injecting them into the prompt — bridging parametric knowledge (what the model learned) with non-parametric knowledge (live data).

```
User Query → Retrieve Docs → Augment Prompt → LLM → Response
```

---

## Core Pipeline

```
┌─────────────┐    ┌──────────────┐    ┌────────────────┐
│  Documents  │───▶│   Chunking   │───▶│   Embedding    │
└─────────────┘    └──────────────┘    └────────────────┘
                                                │
                                                ▼
┌─────────────┐    ┌──────────────┐    ┌────────────────┐
│  LLM Answer │◀───│ Augmented    │◀───│  Vector Store  │
└─────────────┘    │  Prompt      │    └────────────────┘
                   └──────────────┘            ▲
                                               │
                                       ┌───────────────┐
                                       │  User Query   │
                                       │  (Embedded)   │
                                       └───────────────┘
```

---

## Indexing Phase

### 1. Document Loading
| Source | Tools |
|--------|-------|
| PDFs / DOCX | LangChain loaders, PyMuPDF, Unstructured |
| Web pages | BeautifulSoup, Firecrawl, Apify |
| Databases | SQLAlchemy, custom connectors |
| APIs / JSON | Custom parsers |

### 2. Chunking Strategies
| Strategy | Best For | Notes |
|----------|----------|-------|
| **Fixed-size** | Simple docs | Fast, may break context |
| **Recursive split** | General text | Respects sentences/paragraphs |
| **Semantic split** | High-quality retrieval | Groups by meaning |
| **Document-aware** | Structured docs (MD, HTML) | Splits by headers/sections |
| **Sliding window** | Dense info | Overlapping chunks for continuity |

> 💡 **Chunk size rule of thumb:** 256–512 tokens for factual Q&A; 512–1024 for summarization tasks. Always include **20–50 token overlap**.

### 3. Embedding Models
| Model | Dims | Notes |
|-------|------|-------|
| `text-embedding-3-small` | 1536 | OpenAI, fast & cheap |
| `text-embedding-3-large` | 3072 | OpenAI, higher accuracy |
| `nomic-embed-text` | 768 | Open-source, strong performer |
| `bge-m3` | 1024 | Multilingual, SOTA open-source |
| `mxbai-embed-large` | 1024 | Great for retrieval tasks |

---

## Vector Stores
| Store | Type | Best For |
|-------|------|----------|
| **Pinecone** | Managed cloud | Production, scalability |
| **Weaviate** | Hybrid (OSS/cloud) | Hybrid search, multimodal |
| **Qdrant** | OSS / cloud | High performance, filtering |
| **Chroma** | OSS local | Dev/prototyping |
| **pgvector** | PostgreSQL ext. | Existing Postgres stack |
| **FAISS** | In-memory (Meta) | Research, offline use |
| **Milvus** | OSS / cloud | Large-scale production |

---

## Retrieval Phase

### Retrieval Methods
| Method | Description |
|--------|-------------|
| **Dense retrieval** | ANN search on vector embeddings |
| **Sparse retrieval** | BM25 / TF-IDF keyword matching |
| **Hybrid retrieval** | Dense + sparse, re-ranked (best accuracy) |
| **Multi-query** | LLM generates multiple query variants |
| **HyDE** | Hypothetical Document Embeddings — embed a generated answer, then search |
| **Contextual compression** | Filter/compress retrieved chunks before LLM |

### Similarity Metrics
| Metric | Use When |
|--------|----------|
| **Cosine similarity** | Most embedding models (default) |
| **Dot product** | Normalized vectors, faster |
| **Euclidean (L2)** | Spatial distance tasks |

### Re-Ranking
Run a **cross-encoder** on top-k results for better precision before sending to LLM.

```
Query + Doc → Cross-Encoder → Relevance Score → Re-rank → Top-n to LLM
```

Popular re-rankers: `Cohere Rerank`, `bge-reranker`, `ms-marco-MiniLM`

---

## Generation Phase

### Prompt Structure
```
System: You are a helpful assistant. Answer ONLY using the provided context.
        If the answer is not in the context, say "I don't know."

Context:
<retrieved chunk 1>
<retrieved chunk 2>
...

Question: {user_query}

Answer:
```

### Context Window Tips
- Keep retrieved context **under 60–70%** of the context window
- Place most relevant chunks **first** (primacy effect)
- Add **source metadata** (doc title, page, URL) for citations

---

## Advanced RAG Techniques

| Technique | Description |
|-----------|-------------|
| **Parent-child chunking** | Index small chunks, retrieve parent for context |
| **RAPTOR** | Recursive summarization tree for multi-level retrieval |
| **Self-RAG** | LLM decides when/what to retrieve dynamically |
| **CRAG** | Corrective RAG — evaluates retrieval quality, falls back to web search |
| **GraphRAG** | Knowledge graphs for multi-hop reasoning |
| **Agentic RAG** | Agent decides retrieval strategy and tools |
| **Metadata filtering** | Pre-filter by date, category, source before vector search |

---

## Evaluation Metrics

| Metric | Measures |
|--------|----------|
| **Faithfulness** | Is the answer grounded in the retrieved context? |
| **Answer relevancy** | Does the answer address the question? |
| **Context precision** | What % of retrieved chunks are relevant? |
| **Context recall** | Did retrieval capture all needed information? |
| **RAGAS score** | Composite of the above (framework) |

**Tools:** `RAGAS`, `TruLens`, `DeepEval`, `LangSmith`

---

## Common Failure Modes

| Problem | Cause | Fix |
|---------|-------|-----|
| Hallucination | LLM ignores context | Stricter system prompt; lower temperature |
| Low retrieval recall | Bad chunking or embeddings | Tune chunk size; try hybrid search |
| Context stuffing | Too many chunks | Re-rank; use compression |
| Stale data | Index not updated | Schedule re-indexing or use streaming ingestion |
| Query-doc mismatch | User query ≠ document language | Use HyDE or query rewriting |

---

## Quick-Start Stack (Python)

```python
# Minimal RAG with LangChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# 1. Load & chunk
loader = PyPDFLoader("document.pdf")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 2. Embed & store
vectorstore = Chroma.from_documents(chunks, OpenAIEmbeddings())

# 3. Retrieve & generate
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o"),
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4})
)
answer = qa.invoke("What is the refund policy?")
```

---

## Key Libraries & Frameworks

| Tool | Purpose |
|------|---------|
| **LangChain** | End-to-end RAG pipelines |
| **LlamaIndex** | Document indexing & querying |
| **Haystack** | Production RAG pipelines |
| **DSPy** | Programmatic LLM optimization |
| **RAGAS** | RAG evaluation |
| **Unstructured** | Document parsing (PDFs, HTML, etc.) |

---

*Last updated: 2025 · For RAG architecture questions, always benchmark on your own data.*
