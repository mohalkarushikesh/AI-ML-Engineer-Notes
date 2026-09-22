# Langfuse Cheat Sheet

## What is Langfuse?
Langfuse is an open-source LLM observability and analytics platform for tracing, monitoring, evaluation, prompt management, datasets, and experiments.

---

## Installation

```bash
pip install langfuse
```

```bash
npm install langfuse
```

---

## Environment Variables

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-xxxx
LANGFUSE_SECRET_KEY=sk-lf-xxxx
LANGFUSE_HOST=https://cloud.langfuse.com
```

---

## Python Setup

```python
from langfuse import Langfuse

langfuse = Langfuse()
```

## Create a Trace

```python
trace = langfuse.trace(
    name="chatbot",
    user_id="user123",
    session_id="session1"
)
```

## Create a Span

```python
span = trace.span(
    name="retrieve_docs"
)

span.end()
```

## Generation Tracking

```python
generation = trace.generation(
    name="openai-call",
    model="gpt-4o",
    input="Hello"
)

generation.end(output="Hi there")
```

---

## Decorator (Python)

```python
from langfuse.decorators import observe

@observe()
def my_function():
    return "done"
```

---

## OpenAI Integration

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role":"user","content":"Hello"}]
)
```

---

## Scoring / Evaluation

```python
langfuse.score(
    trace_id=trace.id,
    name="quality",
    value=0.95
)
```

---

## Prompt Management

```python
prompt = langfuse.get_prompt("support-agent")

compiled = prompt.compile(
    user_name="Rushikesh"
)
```

---

## Dataset Example

```python
dataset = langfuse.create_dataset(
    name="qa-benchmark"
)
```

---

## Common Concepts

| Concept | Purpose |
|----------|----------|
| Trace | End-to-end request |
| Span | Step within trace |
| Generation | LLM call |
| Score | Evaluation metric |
| Dataset | Test cases |
| Prompt | Versioned prompts |
| Session | User conversation |

---

## Useful Filters

- User ID
- Session ID
- Model Name
- Tags
- Environment
- Trace Status

---

## Best Practices

1. Trace every user request.
2. Create spans for retrieval, tools, and business logic.
3. Log prompt + model output.
4. Add evaluation scores.
5. Use datasets for regression testing.
6. Version prompts through Langfuse.
7. Tag production and staging separately.

---

## Quick Workflow

```text
User Request
    ↓
Trace
    ↓
Span(s)
    ↓
Generation (LLM Call)
    ↓
Score / Evaluation
    ↓
Analytics Dashboard
```
