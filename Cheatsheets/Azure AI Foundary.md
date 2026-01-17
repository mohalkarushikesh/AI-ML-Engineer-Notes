Here’s a **straight, practical Microsoft Foundry (Azure AI Foundry) cheat sheet** — aligned with **enterprise LLMs, RAG, agents, and production AI**.

> Microsoft **rebranded & unified** Azure OpenAI, Prompt Flow, AI Studio, and model catalog under **Azure AI Foundry**.

---

# 🟦 Microsoft Azure AI Foundry Cheat Sheet

---

## 🔹 What Is Microsoft AI Foundry?

**Azure AI Foundry** is Microsoft’s **end-to-end platform** to:

* Build **LLM apps**
* Use **OpenAI + OSS models**
* Create **RAG systems**
* Build **AI agents**
* Monitor, evaluate, and deploy securely

Think of it as **enterprise Hugging Face + LangSmith + OpenAI Studio**.

---

## 🧠 Core Components

| Component           | Purpose                      |
| ------------------- | ---------------------------- |
| **Model Catalog**   | OpenAI, Llama, Mistral, Phi  |
| **Prompt Flow**     | Prompt + chain orchestration |
| **Azure AI Search** | Vector DB for RAG            |
| **AI Studio**       | UI for building/testing      |
| **Evaluation**      | Quality, safety, cost        |
| **Deployment**      | APIs, endpoints              |

---

## 🔹 Model Catalog (Key Models)

* **OpenAI**: GPT-4o, GPT-4.1, GPT-35
* **Microsoft Phi**: Phi-2, Phi-3 (lightweight 🔥)
* **Meta**: Llama 3
* **Mistral**
* **Cohere (embeddings)**

---

## 🔹 Create Project

1. Azure Portal → **Azure AI Foundry**
2. Create **AI Hub**
3. Create **Project**
4. Select models + region

---

## 🔹 Prompt Flow (MOST IMPORTANT 🔥)

### What Is Prompt Flow?

* DAG of prompts + tools + Python
* Versioned
* Testable
* Production ready

Used for:

* Chatbots
* RAG
* Agents
* Multi-step reasoning

---

### Prompt Flow Example

```yaml
inputs:
  question: string

outputs:
  answer: string

nodes:
- name: llm_call
  type: llm
  source:
    model: gpt-4o
    prompt: |
      Answer clearly:
      {{question}}
```

---

## 🔹 Python Prompt Flow

```python
from promptflow import tool

@tool
def clean_input(text: str) -> str:
    return text.strip().lower()
```

---

## 🔹 Azure OpenAI (LLM Call)

```python
from openai import AzureOpenAI

client = AzureOpenAI(
  azure_endpoint=AZURE_ENDPOINT,
  api_key=AZURE_KEY,
  api_version="2024-02-15-preview"
)

response = client.chat.completions.create(
  model="gpt-4o",
  messages=[{"role":"user","content":"Explain RAG"}]
)

print(response.choices[0].message.content)
```

---

## 🔹 Embeddings (RAG)

```python
response = client.embeddings.create(
  model="text-embedding-3-large",
  input="Azure AI Foundry"
)
vector = response.data[0].embedding
```

---

## 🔹 RAG Architecture (Foundry Way)

```
User Query
   ↓
Embedding Model
   ↓
Azure AI Search (Vector DB)
   ↓
Top-K Documents
   ↓
LLM (GPT-4o / Phi)
   ↓
Answer
```

---

## 🔹 Azure AI Search (Vector DB)

* Hybrid search (keyword + vector)
* Enterprise-grade
* Scales easily
* Native Azure integration

Used instead of:

* Pinecone
* FAISS (prod)
* Weaviate

---

## 🔹 Agents (Tool Calling)

```python
tools = [{
  "type": "function",
  "function": {
    "name": "get_weather",
    "parameters": {
      "city": {"type":"string"}
    }
  }
}]
```

LLM decides **when to call tools**.

---

## 🔹 Evaluation & Monitoring

* Prompt regression tests
* Cost tracking
* Latency
* Toxicity & safety
* Human feedback loops

Equivalent to **LangSmith** in Azure ecosystem.

---

## 🔹 Deployment

* REST endpoint
* Azure Functions
* AKS
* App Service
* Secure VNET

---

## 🔹 Security (Enterprise 🔐)

* RBAC
* Private endpoints
* Data never used for training
* SOC2 / ISO / GDPR compliant

This is why **banks + MNCs use it**.

---

## 🔹 Microsoft Foundry vs Others

| Platform             | Best For             |
| -------------------- | -------------------- |
| **Azure AI Foundry** | Enterprise, security |
| Hugging Face         | Research, training   |
| Ollama               | Local/offline        |
| LangSmith            | Observability        |
| OpenAI               | Fast prototyping     |

---

## 🔥 Best Models (Cost vs Power)

| Use Case    | Model               |
| ----------- | ------------------- |
| Chatbot     | GPT-4o              |
| Cheap prod  | GPT-35              |
| Low latency | Phi-3               |
| RAG         | GPT-4o + embeddings |

---

## 🧠 Interview One-Liners

* **Foundry = enterprise LLM platform**
* **Prompt Flow = LLM orchestration**
* **Azure AI Search = vector DB**
* **Phi models = efficient small LLMs**
* **Secure by default**

---

## ⚡ Your Context (Important)

Given your **AI/ML + backend + enterprise focus**, **Microsoft Foundry is HIGHLY relevant** for:

* MNC projects
* Production AI
* Java/Spring backend integration
* Secure RAG systems

---

## 🧭 What Next?

I can give you:

1. **Azure AI Foundry + RAG full project**
2. **Prompt Flow vs LangChain vs LangGraph**
3. **Java Spring Boot + Azure OpenAI**
4. **Foundry interview Q&A**
5. **End-to-end enterprise chatbot architecture**

Just tell me 🔥
