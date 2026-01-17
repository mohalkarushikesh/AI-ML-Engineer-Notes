Here’s a **practical LangSmith cheat sheet** — the observability/debugging/testing platform for LLM apps from the LangChain team.📊 It helps you trace, evaluate, improve, and ship LLM-powered systems reliably. ([LangChain Blog][1])

---

# 🧠 LangSmith Cheat Sheet

---

## 🚀 What Is LangSmith

**LangSmith** is a unified platform that provides:

* **Tracing & observability** for LLM apps
* **Prompt testing & versioning**
* **Evaluation & metrics dashboards**
* **Debugging tools for chains/agents**
* **API + UI + CLI visibility**
  Works *with or without* LangChain code. ([LangChain Blog][1])

---

## 📦 Quick Setup

### 1. Create Account

* Go to **smith.langchain.com** and sign up. ([LangChain Docs][2])

### 2. Get API Key

* **Settings → API Keys → Create**
* Store the key securely (PAT or service key). ([LangChain Docs][2])

### 3. Set Environment Variables

```bash
export LANGSMITH_API_KEY="YOUR_API_KEY"
```

(Optional) set workspace/project if needed. ([LangChain Docs][2])

---

## 🕵️ Core Features

### 🔍 1. **Tracing / Observability**

Track every operation in your LLM app:

* Model inputs & outputs
* Token usage
* Latency & costs
* Tool calls
* Chains/agent steps
* Errors and patterns
  Just set `LANGSMITH_TRACING=true` and send traces to LangSmith. ([LangChain Docs][3])

**With LangChain**:

```python
from langsmith.wrappers import wrap_openai

client = wrap_openai(openai.OpenAI())
```

Now all LLM usage is auto-logged to LangSmith. ([LangSmith Docs][4])

---

### 🧪 2. **Evaluations (Evals)**

Run systematic evaluations:

* Create evaluation datasets
* Use **prebuilt evaluators** (LLM-as-judge, ranking, etc.)
* Compare models & prompts over time
  Works with openevals or custom suites. ([LangChain Docs][5])

---

### ✍️ 3. **Prompt Engineering**

* Create/edit prompts from UI or SDK
* Version control your prompts
* Test variations easily in the Playground
* Produce better responses from your models
  Supports SDK & UI workflows. ([LangChain Docs][6])

---

## 💻 LangSmith CLI – `langsmith-fetch`

Fetch trace data directly from your terminal:

Install:

```bash
pip install langsmith-fetch
```

Examples:

```bash
# Fetch recent traces
langsmith-fetch traces --project-uuid YOUR_UUID

# Fetch last 5
langsmith-fetch traces --project-uuid YOUR_UUID --limit 5

# Save into directory
langsmith-fetch threads ./my-data --limit 50
```

Each trace/thread is exported as JSON — great for offline debugging or building datasets. ([LangChain Docs][7])

---

## 🛠 SDK Quick Examples

### 🐍 Python Prompt Push

```python
from langsmith import Client
from langchain_core.prompts import ChatPromptTemplate

client = Client()

prompt = ChatPromptTemplate([
    ("system", "You are a helpful guy."),
    ("user", "{question}")
])

client.push_prompt("my_prompt", object=prompt, tags=["test"])
```

Prompt will be stored & visible in LangSmith. ([LangSmith Docs][8])

---

## 🧠 Common Env & Flags

| Env Variable             | Purpose                      |
| ------------------------ | ---------------------------- |
| `LANGSMITH_API_KEY`      | Auth                         |
| `LANGSMITH_TRACING`      | Enable trace logging         |
| `LANGSMITH_WORKSPACE_ID` | (Optional) specify workspace |

---

## 📊 Common Workflows

### 🐞 Debugging a Chain

1. Run your LLM app with tracing on
2. Open LangSmith dashboard
3. Inspect trace steps → find the step causing errors
4. Edit prompt/chain → re-run → repeat

### 🧪 Eval Across Models

1. Create dataset of inputs & expected outputs
2. Run multiple models on dataset
3. View evaluator scores in LangSmith UI
4. Save best model/prompt combo

---

## 🧩 Integrations

* Works with LangChain, LangGraph, or raw SDK (Python/TS) apps
* Wrap clients like OpenAI to auto trace
* Can be paired with monitoring stacks plus self-hosting for enterprise ([LangChain][9])

---

## 🧠 Tips & Gotchas

✅ You can **fetch trace data programmatically** via CLI and pipe to tools. ([LangChain Docs][7])
✅ Useful for **regression tests** across prompt versions.
⚠️ Security: watch out for malicious shared agents that may leak keys if you “Try It” from hubs (patched now). ([The Hacker News][10])

---

## 🧾 Summary (One-Liner)

**LangSmith** = observability + prompt management + evaluation + debugging platform for building **production-grade LLM applications** with visibility into every call and outcome. ([LangChain Blog][1])

---

If you want, I can also give you **LangSmith + LangChain code examples** tailored for your Java/ML backend projects.

[1]: https://blog.langchain.com/announcing-langsmith?utm_source=chatgpt.com "Announcing LangSmith, a unified platform for debugging, testing, evaluating, and monitoring your LLM applications"
[2]: https://docs.langchain.com/langsmith/home?utm_source=chatgpt.com "LangSmith docs - Docs by LangChain"
[3]: https://docs.langchain.com/langsmith/trace-openai?utm_source=chatgpt.com "Trace with OpenAI - Docs by LangChain"
[4]: https://docs.smith.langchain.com/reference/python/wrappers/langsmith.wrappers._openai.wrap_openai?utm_source=chatgpt.com "wrap_openai — 🦜️🛠️ LangSmith documentation"
[5]: https://docs.langchain.com/langsmith/prebuilt-evaluators?utm_source=chatgpt.com "How to use prebuilt evaluators - Docs by LangChain"
[6]: https://docs.langchain.com/langsmith/prompt-engineering-quickstart?utm_source=chatgpt.com "Prompt engineering quickstart - Docs by LangChain"
[7]: https://docs.langchain.com/langsmith/langsmith-fetch?utm_source=chatgpt.com "LangSmith Fetch - Docs by LangChain"
[8]: https://docs.smith.langchain.com/prompt_engineering/quickstarts/quickstart_sdk?utm_source=chatgpt.com "Prompt Engineering Quick Start (SDK) | 🦜️🛠️ LangSmith"
[9]: https://www.langchain.com/langsmith?utm_source=chatgpt.com "LangSmith - Observability"
[10]: https://thehackernews.com/2025/06/langchain-langsmith-bug-let-hackers.html?utm_source=chatgpt.com "LangSmith Bug Could Expose OpenAI Keys and User Data via Malicious Agents"
