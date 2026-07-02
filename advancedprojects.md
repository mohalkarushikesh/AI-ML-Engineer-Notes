Here's a set of projects organized the way you framed it, each spanning the full pipeline (data → model/pipeline → evaluation → deployment), with tooling that's actually current as of mid-2026 rather than 2023-era tutorial stacks.

### Machine Learning
- **Real-time fraud/anomaly detector** — streaming features (Kafka), gradient boosting (XGBoost/LightGBM), a feature store (Feast), model registry + CI/CD (MLflow), drift monitoring (Evidently), served via FastAPI with a live dashboard.
- **Multi-SKU demand forecasting platform** — compare classical methods (Prophet) against deep learning ones (Temporal Fusion Transformer via Darts/NeuralForecast), rolling backtests, a Streamlit dashboard for non-technical stakeholders.
- **Credit-risk or churn scorer with explainability built in** — gradient boosting + SHAP, fairness auditing (Fairlearn), and audit-logged predictions so every score is traceable — this governance layer is what separates a portfolio model from a production one.

### Deep Learning
- **Domain-specific detector** — fine-tune YOLO or a ViT-based detector on a niche dataset (defect detection, camera-trap wildlife, satellite imagery), add an active-learning loop to cut labeling cost, export to ONNX for edge deployment.
- **LoRA-tuned generative app** — fine-tune Stable Diffusion or FLUX on a narrow style/product domain, add safety filtering, wrap it in a web front-end.
- **Speech pipeline** — fine-tune Whisper on an accent or domain vocabulary, add speaker diarization, deploy as a real-time transcription service.

### NLP / LLMs
- **PEFT fine-tune a small open model** — LoRA/QLoRA (Hugging Face PEFT + Unsloth for speed) on a domain corpus. Small models like Qwen3.5's 2B/4B variants and SmolLM3 are built specifically for cheap fine-tuning and support a switchable fast-vs-reasoning mode, so they're a good target. Quantize to GGUF, serve locally with Ollama or vLLM, and benchmark against the base model.
- **DPO alignment mini-project** — curate a preference dataset, run DPO via the TRL library, evaluate the aligned model against the base one on a held-out set. This teaches the actual mechanics behind "chat" fine-tuning instead of treating it as a black box.
- **Structured extraction system** — NER/relation extraction over messy domain text (contracts, medical notes, filings) using fine-tuned transformers or LLM output validated against a schema (Pydantic/Instructor), with a human-correction loop feeding back into the training set.

### RAG (this is where 2026 has moved fastest)
- **Adaptive/agentic RAG router** — the current reference architecture. The 2026 state of the art is Adaptive RAG, where a query-complexity classifier decides which retrieval strategy a given question actually needs — simple questions get fast vector search, multi-document ones get hybrid search (dense + BM25) with reranking, relationship questions get routed to a graph, and hard ones get a full agentic loop. Layer in self-correction — a step that grades retrieved evidence and triggers a re-query when confidence is low (the Corrective RAG / CRAG pattern) — and study the core agent patterns behind most production systems: router, ReAct, plan-and-execute, multi-agent retrieval, and self-RAG. Orchestrate with LangGraph, which is now the default for this kind of multi-step agent workflow, and add a RAGAS eval harness plus per-path cost/latency tracking.
- **Multimodal "chat with your PDFs"** — parse documents with layout-aware extraction (Docling/unstructured.io), then embed pages directly using late-interaction vision models like ColPali or ColQwen2, now the standard way to make tables and charts retrievable instead of just body text.
- **GraphRAG over a relationship-heavy corpus** — build a knowledge graph (Neo4j) from unstructured text — biomedical literature, org charts, supply-chain docs — and combine graph traversal with vector search for multi-hop questions plain vector RAG can't answer.
- **Permission-aware enterprise assistant** — RAG with document/row-level access control, inline source citations, and a low-confidence flag that routes answers to human review. This security/governance layer is now treated as core RAG infrastructure, not a bolt-on.

For the retrieval layer itself: Qdrant is the strongest choice if you're self-hosting and care about latency and cost, Pinecone if you want a managed default.

### Agentic AI & new tooling
- **Build your own MCP server** — MCP is now core agent infrastructure, often described as a universal connector for hooking any agent up to any tool, with well over a thousand community servers already available; fastmcp is the go-to Python library for building your own. Expose a real tool (an internal API, a database) and wire it into Claude Code or a custom agent.
- **Multi-agent build/research crew** — planner, retriever, coder, and reviewer sub-agents orchestrated with LangGraph or CrewAI. Wire tool access through MCP and agent-to-agent delegation through the newer A2A protocol, so a planner agent can hand off subtasks to specialist agents the way a real team would. Add human-approval checkpoints before any risky action executes.
- **Browser/computer-use agent** — completes multi-step web tasks (form-filling, data extraction) using a vision-language model for grounding, run inside a sandbox with a required human confirmation before anything irreversible happens.
- **LLM eval/observability platform** — a harness that scores every prompt or model change against a fixed eval set (faithfulness, hallucination rate, latency, cost) and flags regressions automatically. Essentially CI/CD for prompts — most teams running LLMs in production are missing exactly this.

For open models to fine-tune or build agents on top of: Qwen3, DeepSeek R1, Llama 4 Scout, Gemma 3, and Phi-4 and Kimi K2.6, DeepSeek V4 Pro, GLM-5, and Qwen3.6 currently anchor the open-weight frontier, each strong in a different lane (reasoning, long context, coding, edge deployment) — worth picking based on what your project actually needs rather than defaulting to the biggest name.

---

Pick any one of these and I can turn it into a real build guide — architecture, repo structure, and a week-by-week plan — or narrow the list further if you tell me your current skill level or whether this is for a portfolio, work, or research.Once you've settled on one, **Claude Code** is worth using for the actual build — it can scaffold the repo, write the training/RAG/agent pipeline code, and iterate with you end to end rather than you copy-pasting snippets from chat.
