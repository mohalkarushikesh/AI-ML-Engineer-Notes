# LangSmith Cheatsheet

> Observability, testing, and evaluation platform for LLM applications built with LangChain (or any framework).

---

## Installation & Setup

```bash
pip install langsmith
pip install langchain-langsmith   # LangChain integration
```

```python
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"]    = "ls__..."
os.environ["LANGCHAIN_PROJECT"]    = "my-project"   # optional, default: "default"
os.environ["LANGCHAIN_ENDPOINT"]   = "https://api.smith.langchain.com"  # default
```

---

## Tracing

### Auto-tracing (LangChain)

Set the env vars above — all LangChain/LCEL runs are traced automatically with zero code changes.

### `@traceable` decorator (any framework)

```python
from langsmith import traceable

@traceable
def my_llm_call(prompt: str) -> str:
    # works with OpenAI, Anthropic, or any SDK
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

result = my_llm_call("What is LangSmith?")
```

### `@traceable` with metadata

```python
@traceable(
    name="my_pipeline",          # display name in UI
    tags=["production", "v2"],
    metadata={"user_id": "u123"},
    run_type="chain",            # "chain" | "llm" | "tool" | "retriever" | "embedding"
)
def pipeline(query: str) -> str:
    ...
```

### Context manager (manual spans)

```python
from langsmith import trace

with trace("my_span", run_type="chain", tags=["debug"]) as run:
    result = do_something()
    run.end(outputs={"result": result})
```

### Wrap OpenAI client

```python
from langsmith.wrappers import wrap_openai
import openai

client = wrap_openai(openai.Client())
# All calls through `client` are now traced automatically
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
)
```

### Wrap Anthropic client

```python
from langsmith.wrappers import wrap_anthropic
import anthropic

client = wrap_anthropic(anthropic.Anthropic())
```

### Disable tracing temporarily

```python
from langsmith import utils
utils.tracing_is_enabled()   # check status

# Disable for a block
import os
os.environ["LANGCHAIN_TRACING_V2"] = "false"
```

---

## LangSmith Client

```python
from langsmith import Client

client = Client(
    api_key="ls__...",
    api_url="https://api.smith.langchain.com",  # default
)
```

---

## Projects

```python
# List projects
projects = client.list_projects()
for p in projects:
    print(p.name, p.id)

# Create project
client.create_project("my-new-project", description="RAG pipeline evals")

# Delete project
client.delete_project(project_name="old-project")
```

---

## Runs

```python
# List runs in a project
runs = client.list_runs(
    project_name="my-project",
    run_type="chain",               # filter by type
    filter='eq(status, "error")',   # filter expression
    limit=50,
)

for run in runs:
    print(run.id, run.name, run.status, run.total_tokens)

# Get a specific run
run = client.read_run(run_id="<uuid>")

# Get run URL
url = client.get_run_url(run=run)

# Share a run (create public link)
share_token = client.share_run(run_id="<uuid>")

# Update run (add feedback programmatically)
client.update_run(run_id="<uuid>", extra={"custom_key": "value"})
```

### Run Filter Expressions

```python
# Runs with errors
'eq(status, "error")'

# Runs longer than 5 seconds
'gt(latency, 5)'

# Runs with a specific tag
'has(tags, "production")'

# Runs in a date range
'and(gte(start_time, "2024-01-01"), lte(start_time, "2024-12-31"))'

# Runs containing a keyword in input
'like(inputs.question, "%langsmith%")'
```

---

## Feedback

```python
# Add feedback to a run
client.create_feedback(
    run_id="<uuid>",
    key="correctness",          # feedback key
    score=1,                    # numeric score (0–1 recommended)
    comment="Great answer!",
    value="correct",            # optional string label
)

# Thumbs up / down
client.create_feedback(run_id="<uuid>", key="thumbs", score=1)
client.create_feedback(run_id="<uuid>", key="thumbs", score=0)

# List feedback for a run
feedback = client.list_feedback(run_ids=["<uuid>"])

# Feedback keys in use
keys = client.list_feedback_definitions()
```

---

## Datasets

```python
# Create a dataset
dataset = client.create_dataset(
    dataset_name="qa-pairs",
    description="Question-answer evaluation set",
    data_type="kv",   # "kv" | "llm" | "chat"
)

# Add examples
client.create_examples(
    inputs=[
        {"question": "What is LangSmith?"},
        {"question": "What is LangChain?"},
    ],
    outputs=[
        {"answer": "LangSmith is an observability platform."},
        {"answer": "LangChain is an LLM application framework."},
    ],
    dataset_id=dataset.id,
)

# Add single example
client.create_example(
    inputs={"question": "Capital of France?"},
    outputs={"answer": "Paris"},
    dataset_name="qa-pairs",
)

# List datasets
datasets = client.list_datasets()

# Read a dataset
dataset = client.read_dataset(dataset_name="qa-pairs")

# List examples in a dataset
examples = list(client.list_examples(dataset_name="qa-pairs"))

# Delete a dataset
client.delete_dataset(dataset_name="old-dataset")

# Create dataset from existing runs
dataset = client.create_dataset("from-runs")
client.create_examples(
    inputs=[run.inputs for run in runs],
    outputs=[run.outputs for run in runs],
    dataset_id=dataset.id,
)
```

### Upload CSV as dataset

```python
import pandas as pd

df = pd.read_csv("eval_data.csv")
client.upload_dataframe(
    df,
    name="csv-dataset",
    input_keys=["question"],
    output_keys=["answer"],
)
```

---

## Evaluation

### `evaluate()` — standard eval

```python
from langsmith.evaluation import evaluate

def my_app(inputs: dict) -> dict:
    answer = chain.invoke(inputs["question"])
    return {"answer": answer}

def correctness_evaluator(run, example) -> dict:
    predicted = run.outputs["answer"]
    expected  = example.outputs["answer"]
    score = 1 if expected.lower() in predicted.lower() else 0
    return {"key": "correctness", "score": score}

results = evaluate(
    my_app,
    data="qa-pairs",               # dataset name or id
    evaluators=[correctness_evaluator],
    experiment_prefix="baseline",  # optional label
    num_repetitions=1,
    max_concurrency=4,
)

print(results.to_pandas())
```

### LLM-as-Judge evaluator

```python
from langsmith.evaluation import LangChainStringEvaluator

# Built-in criteria evaluators
criteria_eval = LangChainStringEvaluator(
    "criteria",
    config={
        "criteria": {
            "helpfulness": "Is the response helpful and informative?",
        }
    }
)

qa_eval = LangChainStringEvaluator("qa")           # checks answer correctness
cot_qa_eval = LangChainStringEvaluator("cot_qa")   # chain-of-thought QA eval

results = evaluate(
    my_app,
    data="qa-pairs",
    evaluators=[criteria_eval, qa_eval],
)
```

### Custom LLM evaluator

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

def llm_evaluator(run, example) -> dict:
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = ChatPromptTemplate.from_template(
        "Rate the answer 0-1 for correctness.\n"
        "Question: {question}\nAnswer: {answer}\nExpected: {expected}\n"
        "Return only a number."
    )
    chain = prompt | llm
    result = chain.invoke({
        "question": example.inputs["question"],
        "answer": run.outputs["answer"],
        "expected": example.outputs["answer"],
    })
    score = float(result.content.strip())
    return {"key": "llm_correctness", "score": score}
```

### Compare experiments

```python
# View in UI: Projects → Experiments → Compare
# Or programmatically:
results_a = evaluate(my_app_v1, data="qa-pairs", experiment_prefix="v1")
results_b = evaluate(my_app_v2, data="qa-pairs", experiment_prefix="v2")
```

---

## Online Evaluation (Auto-feedback on traces)

```python
# Define an evaluator that runs automatically on new traces
# Set up in LangSmith UI: Projects → Automations → Add Evaluator
# Or via API:

from langsmith.schemas import RunEvaluatorResult
from langsmith.evaluation import RunEvaluator

class MyOnlineEvaluator(RunEvaluator):
    def evaluate_run(self, run, example=None) -> RunEvaluatorResult:
        score = 1 if "sorry" not in run.outputs.get("output", "") else 0
        return RunEvaluatorResult(
            key="no_apology",
            score=score,
        )
```

---

## Prompt Hub

```python
from langchain import hub

# Pull a prompt
prompt = hub.pull("rlm/rag-prompt")

# Pull specific version
prompt = hub.pull("rlm/rag-prompt:abc123")

# Push a prompt
hub.push("my-username/my-prompt", prompt, new_repo_is_public=False)

# List available prompts (UI: smith.langchain.com/hub)
```

---

## Annotations & Human Review

```python
# Create an annotation queue in the UI or API
queue = client.create_annotation_queue(
    name="human-review",
    description="Manual review of edge cases",
)

# Add runs to queue
client.add_runs_to_annotation_queue(
    queue_id=queue.id,
    run_ids=["<uuid1>", "<uuid2>"],
)

# List queues
queues = client.list_annotation_queues()
```

---

## Monitoring & Metrics

Access in LangSmith UI under **Projects → Monitor**:

| Metric | Description |
|--------|-------------|
| Latency (p50/p99) | Response time percentiles |
| Token usage | Input / output / total tokens |
| Error rate | % of failed runs |
| Feedback scores | Average score per feedback key |
| Run volume | Traces per hour/day |

```python
# Fetch aggregate stats programmatically
stats = client.get_run_stats(
    project_name="my-project",
    run_type="chain",
    start_time="2024-01-01",
)
```

---

## Async Support

```python
from langsmith import AsyncClient

client = AsyncClient()

async def main():
    run = await client.aread_run(run_id="<uuid>")
    await client.acreate_feedback(run_id="<uuid>", key="score", score=1)
    results = [r async for r in client.alist_runs(project_name="my-project")]
```

---

## Environment Variable Reference

| Variable | Description | Default |
|----------|-------------|---------|
| `LANGCHAIN_TRACING_V2` | Enable tracing | `false` |
| `LANGCHAIN_API_KEY` | LangSmith API key | — |
| `LANGCHAIN_PROJECT` | Project name | `"default"` |
| `LANGCHAIN_ENDPOINT` | API endpoint | `https://api.smith.langchain.com` |
| `LANGCHAIN_HIDE_INPUTS` | Redact run inputs | `false` |
| `LANGCHAIN_HIDE_OUTPUTS` | Redact run outputs | `false` |

---

## Self-Hosted (LangSmith Server)

```bash
# Pull and run with Docker Compose
git clone https://github.com/langchain-ai/langsmith-sdk
cd langsmith-sdk/ops
docker compose up

# Point client to local instance
os.environ["LANGCHAIN_ENDPOINT"] = "http://localhost:1984"
```

---

## Tips & Gotchas

- Set `LANGCHAIN_PROJECT` per environment (`dev`, `staging`, `production`) to keep traces organized.
- Use `@traceable` on non-LangChain code (raw OpenAI, Anthropic, custom pipelines) to get full visibility.
- `wrap_openai()` / `wrap_anthropic()` is the easiest way to trace SDK calls without changing business logic.
- `evaluate()` runs examples concurrently — set `max_concurrency` to avoid rate limits.
- Feedback `score` should be numeric (0–1) for aggregation to work in the monitoring dashboard.
- Dataset examples are immutable once created; delete and re-add to update them.
- Use `experiment_prefix` in `evaluate()` to label runs meaningfully before comparing in the UI.
- Tracing adds ~1–5 ms latency per run; use `LANGCHAIN_TRACING_V2=false` in latency-critical paths.
- LangSmith stores traces for 14 days on the free plan; upgrade for longer retention.
