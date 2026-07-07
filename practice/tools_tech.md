Here are hands-on practice exercises for the modern AI/LLM engineering stack, organized by level. I've grouped the tools you named alongside the others they're usually paired with (vector DBs, LlamaIndex, Hugging Face, FastAPI, Streamlit/Gradio, agent frameworks, fine-tuning tools, and observability platforms). Each item is a mini-project you can build and run.

## Beginner

1. **Run a local LLM with Ollama** — Install Ollama, pull a few models (e.g., Llama, Mistral, Phi), and chat with them from the CLI and the REST API; compare speed and quality across model sizes.
2. **First LangChain chain** — Build a simple prompt → LLM → output-parser chain; swap between an Ollama model and a hosted API to see the abstraction in action.
3. **Prompt templates & output parsing** — Use LangChain prompt templates with variables, then parse the response into structured data with Pydantic (structured output).
4. **Streamlit/Gradio chat UI** — Wrap any LLM in a simple chat interface so you have a reusable front-end for later projects.
5. **Embeddings 101** — Generate embeddings for a set of sentences (via Hugging Face `sentence-transformers` or Ollama), compute cosine similarity, and find the nearest neighbors.
6. **Your first vector store** — Load documents into Chroma or FAISS, embed them, and run a similarity search query.
7. **Dockerize a Python app** — Write a Dockerfile for a small Python script, build the image, and run it in a container; understand images vs. containers, layers, and volumes.
8. **Hugging Face pipelines** — Use the `transformers` pipeline API for sentiment, summarization, and NER without any training, then load a model manually to see what the pipeline hides.

## Medium

1. **RAG pipeline end-to-end** — Chunk documents, embed them into a vector DB (Chroma/Qdrant/pgvector), retrieve relevant chunks, and feed them to an LLM to answer questions with citations.
2. **Compare RAG frameworks** — Build the same RAG app once in LangChain and once in LlamaIndex; compare the developer experience and retrieval quality.
3. **Add LangSmith tracing** — Instrument a LangChain/LangGraph app with LangSmith to trace every step, inspect token usage and latency, and debug a failing chain.
4. **Build an agent with tools** — Create an agent that can call tools (calculator, web search, a custom function) and reason about which to use; observe the tool-calling loop.
5. **First LangGraph workflow** — Rebuild an agent as a stateful graph with nodes and edges; add branching logic and a loop, and visualize the graph.
6. **Build an MCP server** — Write a simple MCP server that exposes a tool or data source (e.g., a file reader or a small API), then connect it to an MCP-compatible client and call it.
7. **Structured extraction service** — Use Pydantic + an LLM (via Instructor or LangChain) to reliably extract structured JSON from unstructured text, with validation and retries.
8. **Serve a model with FastAPI + Docker** — Wrap an LLM or an embedding endpoint in a FastAPI service, containerize it, and test it with requests.
9. **Docker Compose multi-service app** — Run a vector database, your API, and a UI together with `docker-compose`; connect them over the Docker network.
10. **Evaluate an LLM app** — Build a small evaluation set and use LangSmith (or a simple custom harness) to score outputs on correctness, relevance, and hallucination.
11. **Conversational memory** — Add short-term and summary-based memory to a chatbot so it remembers earlier turns; test where naive memory breaks down.

## Advanced

1. **Multi-agent system** — Use LangGraph (or CrewAI/AutoGen) to build multiple agents that collaborate — e.g., a researcher, a writer, and a critic — with a shared state and hand-offs.
2. **Agentic RAG** — Build a graph where the agent decides *whether* to retrieve, reformulates queries, grades retrieved documents, and re-retrieves if they're irrelevant (self-correcting RAG).
3. **Production MCP integration** — Build an MCP server exposing several real tools (database queries, file operations, an external API), add auth, and integrate it into an agent that orchestrates them.
4. **Human-in-the-loop workflows** — Use LangGraph's persistence and interrupt features to pause a graph for human approval before an agent takes a consequential action, then resume.
5. **Fine-tune a model with LoRA/PEFT** — Fine-tune an open model (via Hugging Face PEFT, Unsloth, or Axolotl) on a custom instruction dataset; quantize it and run it locally through Ollama.
6. **Self-hosted high-throughput serving** — Serve an open model with vLLM or Hugging Face TGI for batched, high-throughput inference; benchmark tokens/sec against Ollama.
7. **Full observability & eval loop** — Combine LangSmith (or Langfuse / Arize Phoenix) with automated evals in CI so every prompt or code change is regression-tested against a dataset.
8. **Advanced retrieval techniques** — Implement hybrid search (keyword + vector), reranking (cross-encoder), and query expansion; measure the retrieval-quality improvement over naive RAG.
9. **Deploy on Kubernetes** — Take your Dockerized app and deploy it to a Kubernetes cluster (even a local one via kind/minikube) with scaling, health checks, and config management.
10. **Guardrails & safety layer** — Add input/output validation, PII redaction, and content filtering (e.g., via Guardrails AI or custom validators) around an LLM app.
11. **Caching & cost optimization** — Add semantic caching, prompt-level caching, and model routing (cheap model first, escalate to a stronger one) to cut latency and cost; measure the savings.
12. **End-to-end LLM product** — Combine everything: LangGraph agent + MCP tools + RAG over a vector DB + FastAPI backend + Streamlit frontend + LangSmith observability, all Dockerized and deployed. Add CI/CD and an evaluation gate.

A good approach is to complete one project per level fully (design → build → instrument with tracing → evaluate → containerize → short write-up) before advancing. Two habits matter most in this stack: **always add observability early** (you cannot debug an agent you can't see inside), and **always build an evaluation set** (LLM apps silently regress, and vibes-based testing won't catch it).

One caveat: this ecosystem moves extremely fast, and APIs for LangChain, LangGraph, MCP, and the serving tools change often. Before building, check each tool's current official docs for the latest syntax — I can search for the newest documentation or a specific tool's current state if you'd like.

Want me to expand any single exercise into a full step-by-step project with starter code, the current recommended libraries, and a dataset or model suggestion?
