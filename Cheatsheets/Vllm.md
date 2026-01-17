Here’s a **clear, production-focused vLLM cheat sheet** — this is **inference at scale**, not training.

---

# ⚡ vLLM Cheat Sheet (High-Performance LLM Inference)

---

## 🔹 What Is vLLM?

**vLLM** is a **fast LLM inference engine** designed for:

* **Very high throughput**
* **Low latency**
* **Serving many users at once**

🔥 Used by OpenAI-style APIs, startups, and enterprises.

### Core innovation:

**PagedAttention** → efficient KV-cache management
(no wasted GPU memory)

---

## 🧠 When to Use vLLM

Use vLLM when:

* You need **API serving**, not training
* Many concurrent users
* Large context windows
* GPU memory is precious

❌ Not for fine-tuning
❌ Not for CPU-only laptops

---

## 🔹 Install

```bash
pip install vllm
```

GPU required (CUDA).

---

## 🔹 Run a Model (CLI – OpenAI Compatible)

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3-8B-Instruct \
  --host 0.0.0.0 \
  --port 8000
```

Now you have:

```
http://localhost:8000/v1/chat/completions
```

---

## 🔹 Call vLLM API (OpenAI Style)

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3-8B-Instruct",
    "messages": [
      {"role": "user", "content": "Explain RAG"}
    ]
  }'
```

👉 Drop-in replacement for OpenAI API.

---

## 🔹 Python Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

resp = client.chat.completions.create(
    model="meta-llama/Llama-3-8B-Instruct",
    messages=[{"role":"user","content":"Explain transformers"}],
    max_tokens=200
)

print(resp.choices[0].message.content)
```

---

## 🔹 Direct vLLM Usage (No API)

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3-8B-Instruct")

params = SamplingParams(
    temperature=0.7,
    max_tokens=200
)

outputs = llm.generate(
    ["Explain attention mechanism"],
    params
)

print(outputs[0].outputs[0].text)
```

---

## 🔹 Key Flags (IMPORTANT 🔥)

```bash
--tensor-parallel-size 2   # multi-GPU
--gpu-memory-utilization 0.9
--max-model-len 8192
--dtype float16
--trust-remote-code
```

---

## 🔹 Quantization Support

* AWQ
* GPTQ
* FP16 / BF16

Example:

```bash
--quantization awq
```

---

## 🔹 Supported Models

* Llama 2 / 3
* Mistral
* Mixtral
* Qwen
* DeepSeek
* Phi (limited)

Check compatibility before deploying.

---

## 🔹 Performance Comparison

| Engine       | Throughput  | Concurrency |
| ------------ | ----------- | ----------- |
| Transformers | ❌ low       | ❌           |
| Ollama       | ⚠️ medium   | ⚠️          |
| **vLLM**     | ✅ very high | ✅           |
| TGI          | ✅ high      | ✅           |

---

## 🔹 RAG + vLLM (Production Pattern)

```
User
 ↓
Embedding Model
 ↓
Vector DB (FAISS / Azure AI Search)
 ↓
Context Injection
 ↓
vLLM API
 ↓
Answer
```

vLLM only handles **generation**.

---

## 🔹 Scaling

* Horizontal pods (K8s)
* Multi-GPU tensor parallel
* Load balancers
* Auto-scaling APIs

---

## 🔹 Logging & Monitoring

* Request latency
* Tokens/sec
* GPU memory
* Throughput per worker

Usually paired with:

* Prometheus
* Grafana
* LangSmith (eval)

---

## 🔹 vLLM vs Others

| Tool          | Best Use                 |
| ------------- | ------------------------ |
| vLLM          | High-scale inference     |
| Ollama        | Local dev                |
| Hugging Face  | Training + R&D           |
| Azure Foundry | Enterprise orchestration |
| TGI           | Enterprise inference     |

---

## 🔥 Interview Gold Lines

* “vLLM uses **PagedAttention** to optimize KV cache.”
* “It’s **OpenAI-API compatible**.”
* “Designed for **multi-tenant inference**.”
* “Inference-only, not training.”

---

## 🧠 One-Line Summary

**vLLM is the fastest open-source engine for serving LLMs at scale.**

---

## 🧭 What Next?

I can give you:

1. **vLLM vs TGI deep comparison**
2. **vLLM + RAG full prod architecture**
3. **Kubernetes deployment YAML**
4. **Cost optimization tricks**
5. **Java/Spring Boot calling vLLM**

Just say 🔥
