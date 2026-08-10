# Types of Retrieval Augmented Generation (RAG)

RAG combines a **retriever** (fetches relevant external knowledge) with a **generator** (an LLM that produces the answer). The goal is to ground responses in up-to-date, domain-specific, or verifiable data instead of relying only on the model's parametric memory.

**Core pipeline:** `Query → Retrieve → Augment (add context to prompt) → Generate`

---

## 1. Naive / Standard RAG
The baseline approach and starting point for most systems.
- **Flow:** Documents are chunked → embedded into vectors → stored in a vector DB. At query time, the query is embedded, top-k similar chunks are retrieved and stuffed into the prompt.
- **Retrieval:** Simple dense similarity search (e.g., cosine similarity).
- **Strengths:** Easy to build, works for straightforward Q&A.
- **Weaknesses:** Poor recall/precision, retrieves irrelevant chunks, no verification, struggles with complex or multi-hop questions, sensitive to chunking strategy.

## 2. Advanced RAG
Adds optimization steps *around* retrieval. Grouped into pre- and post-retrieval techniques.
- **Pre-retrieval:** query rewriting/expansion, query routing, better chunking (semantic, sliding window, hierarchical), metadata filtering.
- **Post-retrieval:** re-ranking (cross-encoder or LLM), context compression/filtering, context ordering (mitigates "lost in the middle").

## 3. Modular RAG
Treats RAG as interchangeable, reconfigurable modules (search, memory, routing, fusion) rather than a fixed pipeline. Supports iterative and adaptive flows. Highly customizable but more engineering complexity.

## 4. Hybrid RAG
Combines **dense retrieval** (semantic embeddings) with **sparse retrieval** (keyword-based, e.g., BM25). Dense captures meaning; sparse captures exact terms, names, codes. Results merged via Reciprocal Rank Fusion (RRF). Robust across query types.

## 5. Graph RAG
Uses a **knowledge graph** (entities + relationships) instead of, or alongside, vector search. Retrieval traverses the graph to gather connected context. Excellent for multi-hop reasoning and global/summary questions. Higher upfront cost. (Microsoft's *GraphRAG* is a well-known implementation.)

## 6. HyDE (Hypothetical Document Embeddings)
The LLM first generates a *hypothetical answer*, then embeds that and uses it for retrieval — since a hypothetical answer often sits closer to real relevant documents than the raw question. Improves retrieval for vague queries.

## 7. Corrective RAG (CRAG)
Adds a **retrieval evaluator** that grades documents before generation. Correct → refine and use; incorrect/ambiguous → trigger a web search or fallback. Self-correcting, reduces hallucination from bad retrieval.

## 8. Self-RAG
The LLM decides **when** to retrieve and **critiques** its own output using special "reflection tokens." Adaptive (skips retrieval when unnecessary), improves factuality and citations. Needs a model tuned to emit reflection tokens.

## 9. Adaptive RAG
Dynamically chooses a retrieval strategy based on **query complexity**: simple queries answered directly, moderate use single-step retrieval, complex use multi-step/iterative. Balances cost and accuracy.

## 10. Agentic RAG
An **LLM agent** orchestrates retrieval using tools, planning, and reasoning loops — decomposing questions, picking sources, retrieving iteratively, verifying, and synthesizing. Most flexible; higher latency, cost, and complexity.

## 11. Other notable variants

| Type | Key idea |
|------|----------|
| Iterative / Recursive RAG | Retrieve → generate → re-retrieve using intermediate results, in a loop. |
| Multimodal RAG | Retrieves across text, images, tables, audio, or video. |
| Long RAG | Retrieves larger units (sections/documents) to preserve context. |
| RAG-Fusion | Multiple query variants → retrieve for each → fuse with RRF. |
| Speculative / Cache-augmented | Uses caching or preloaded context to cut latency. |

---

## Quick comparison

| Type | Best for | Complexity |
|------|----------|-----------|
| Naive RAG | Simple Q&A, prototypes | Low |
| Advanced RAG | Better precision/recall | Medium |
| Hybrid RAG | Mixed keyword + semantic queries | Medium |
| Graph RAG | Multi-hop, relationship reasoning | High |
| HyDE | Vague queries | Low–Medium |
| Corrective RAG | Reducing bad-retrieval hallucination | Medium |
| Self-RAG | Adaptive retrieval + self-check | High |
| Adaptive RAG | Cost/accuracy balance | Medium–High |
| Agentic RAG | Complex, multi-step, multi-source tasks | High |

---

## Key takeaways
- Start with **Naive RAG**, then add **Advanced** techniques (re-ranking, better chunking, hybrid search) — biggest wins for least effort.
- Move to **Graph, Self, Corrective, Adaptive, or Agentic RAG** as questions get more complex or reliability matters more.
- Most production systems are **hybrids** of several ideas above rather than one "pure" type.
