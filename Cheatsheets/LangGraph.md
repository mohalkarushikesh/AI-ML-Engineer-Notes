# LangGraph Cheatsheet

> Build stateful, multi-step, and multi-agent LLM applications as controllable graphs.

---

## Installation

```bash
pip install langgraph
pip install langchain-openai   # or any LLM provider
```

---

## Core Concepts

| Concept | Description |
|---------|-------------|
| **State** | A typed dict shared across all nodes in the graph |
| **Node** | A Python function that reads and updates state |
| **Edge** | Connection between nodes (static or conditional) |
| **Graph** | The compiled runnable — invoke it like a chain |
| **Checkpointer** | Persists state across steps (enables memory & resume) |

---

## Minimal Example

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

# 1. Define state
class State(TypedDict):
    input: str
    output: str

# 2. Define nodes
def process(state: State) -> State:
    return {"output": state["input"].upper()}

# 3. Build graph
graph = StateGraph(State)
graph.add_node("process", process)
graph.set_entry_point("process")
graph.add_edge("process", END)

# 4. Compile and run
app = graph.compile()
result = app.invoke({"input": "hello"})
print(result["output"])  # HELLO
```

---

## State

### Basic TypedDict state

```python
from typing import TypedDict

class State(TypedDict):
    question: str
    context: str
    answer: str
    steps: int
```

### Reducer — accumulate instead of overwrite

```python
from typing import Annotated
import operator

class State(TypedDict):
    messages: Annotated[list, operator.add]   # appends to list
    count:    Annotated[int, operator.add]    # increments
    result:   str                             # overwrites (default)
```

### MessagesState (built-in for chat)

```python
from langgraph.graph import MessagesState

# Equivalent to:
# class State(TypedDict):
#     messages: Annotated[list[AnyMessage], add_messages]

class State(MessagesState):
    summary: str   # add extra fields
```

### `add_messages` reducer

```python
from langgraph.graph.message import add_messages
from typing import Annotated
from langchain_core.messages import AnyMessage

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    # add_messages: appends new messages, deduplicates by id
```

---

## Nodes

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

llm = ChatOpenAI(model="gpt-4o")

# Simple node — return only the keys you want to update
def call_llm(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def summarize(state: State) -> dict:
    summary = llm.invoke([SystemMessage("Summarize:"), *state["messages"]])
    return {"summary": summary.content}

# Node with tools
llm_with_tools = llm.bind_tools(tools)

def agent_node(state: State) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# Adding nodes
graph.add_node("call_llm", call_llm)
graph.add_node("summarize", summarize)
```

---

## Edges

```python
# Static edge — always goes A → B
graph.add_edge("node_a", "node_b")

# Entry point
graph.set_entry_point("node_a")
# Equivalent to:
graph.add_edge(START, "node_a")

# End
graph.add_edge("node_b", END)

# Conditional edge
def route(state: State) -> str:
    if state["messages"][-1].tool_calls:
        return "tools"
    return END

graph.add_conditional_edges(
    "agent",          # source node
    route,            # routing function → returns node name
    {                 # optional explicit map (for clarity)
        "tools": "tool_node",
        END: END,
    }
)
```

---

## START and END

```python
from langgraph.graph import START, END

graph.add_edge(START, "first_node")
graph.add_edge("last_node", END)
```

---

## Compiling

```python
app = graph.compile()

# With checkpointer (enables memory)
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

# With interrupt (human-in-the-loop)
app = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["human_review"],   # pause before these nodes
    interrupt_after=["tool_node"],       # pause after these nodes
)
```

---

## Invoking the Graph

```python
# Single invocation
result = app.invoke({"messages": [("human", "Hello!")]})

# Stream node outputs
for step in app.stream({"messages": [("human", "Hello!")]}):
    print(step)   # dict: {node_name: state_update}

# Stream tokens
for chunk in app.stream(
    {"messages": [("human", "Hello!")]},
    stream_mode="messages",
):
    print(chunk[0].content, end="")

# With config (required when using checkpointer)
config = {"configurable": {"thread_id": "user_123"}}
result = app.invoke({"messages": [("human", "Hi")]}, config=config)

# Async
result = await app.ainvoke({"messages": [("human", "Hi")]})
async for step in app.astream({"messages": [("human", "Hi")]}):
    print(step)
```

---

## Checkpointers (Persistence & Memory)

```python
# In-memory (dev/testing)
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()

# SQLite (local persistent)
from langgraph.checkpoint.sqlite import SqliteSaver
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")

# PostgreSQL (production)
from langgraph.checkpoint.postgres import PostgresSaver
checkpointer = PostgresSaver.from_conn_string("postgresql://user:pass@host/db")

# Usage — thread_id isolates conversations
config = {"configurable": {"thread_id": "session_42"}}
app.invoke({"messages": [("human", "Remember me")]}, config=config)
app.invoke({"messages": [("human", "What's my name?")]}, config=config)
# The second call has access to the full prior conversation

# Inspect checkpoint state
state = app.get_state(config)
print(state.values)          # current state dict
print(state.next)            # next nodes to execute
print(state.config)          # checkpoint config

# List history
history = list(app.get_state_history(config))
for snapshot in history:
    print(snapshot.config, snapshot.values)
```

---

## Human-in-the-Loop

```python
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["human_review"],
)

config = {"configurable": {"thread_id": "t1"}}

# Run until interrupt
app.invoke({"input": "Do something risky"}, config=config)

# Inspect paused state
state = app.get_state(config)
print("Pending:", state.next)

# Resume (optionally update state first)
app.update_state(config, {"approved": True})
app.invoke(None, config=config)   # None = resume from checkpoint
```

---

## Tool Node (Built-in)

```python
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"

tools = [search]
tool_node = ToolNode(tools)

# Standard ReAct pattern
graph = StateGraph(MessagesState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", tools_condition)
graph.add_edge("tools", "agent")
app = graph.compile()
```

### `tools_condition` routing

```python
# tools_condition returns:
# "tools"  — if last message has tool_calls
# END      — otherwise
graph.add_conditional_edges("agent", tools_condition)
```

---

## Prebuilt ReAct Agent

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    model=llm,
    tools=tools,
    state_modifier="You are a helpful assistant.",  # system prompt
    checkpointer=MemorySaver(),
)

result = agent.invoke(
    {"messages": [("human", "Search for LangGraph news")]},
    config={"configurable": {"thread_id": "s1"}},
)

# Stream
for step in agent.stream(
    {"messages": [("human", "What's the weather in Paris?")]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()
```

---

## Subgraphs

```python
# Define a subgraph
sub = StateGraph(SubState)
sub.add_node("sub_node", sub_node_fn)
sub.set_entry_point("sub_node")
sub.add_edge("sub_node", END)
sub_app = sub.compile()

# Add subgraph as a node in parent graph
parent = StateGraph(ParentState)
parent.add_node("subgraph", sub_app)   # compiled subgraph as node
parent.add_node("other", other_fn)
parent.set_entry_point("subgraph")
parent.add_edge("subgraph", "other")
parent.add_edge("other", END)
```

---

## Multi-Agent (Supervisor Pattern)

```python
from langchain_core.prompts import ChatPromptTemplate

# Each agent is a node
def researcher(state):
    result = researcher_chain.invoke(state)
    return {"messages": [result], "next": "writer"}

def writer(state):
    result = writer_chain.invoke(state)
    return {"messages": [result], "next": "END"}

def supervisor(state):
    decision = supervisor_chain.invoke(state)
    return {"next": decision.next}   # routes to researcher | writer | END

# Route based on supervisor decision
def route_by_next(state):
    return state["next"]

graph = StateGraph(State)
graph.add_node("supervisor", supervisor)
graph.add_node("researcher", researcher)
graph.add_node("writer", writer)
graph.set_entry_point("supervisor")
graph.add_conditional_edges("supervisor", route_by_next)
graph.add_edge("researcher", "supervisor")
graph.add_edge("writer", "supervisor")
```

---

## Parallel Nodes (Fan-out / Fan-in)

```python
# Fan-out: one node → multiple nodes in parallel
graph.add_edge("start_node", "branch_a")
graph.add_edge("start_node", "branch_b")

# Fan-in: multiple nodes → one node
graph.add_edge("branch_a", "merge_node")
graph.add_edge("branch_b", "merge_node")

# State must use reducers to collect parallel results
class State(TypedDict):
    results: Annotated[list, operator.add]   # each branch appends
```

---

## Send API (Dynamic Fan-out)

```python
from langgraph.types import Send

# Dynamically spawn nodes based on state
def fan_out(state: State):
    return [Send("process_item", {"item": item}) for item in state["items"]]

graph.add_conditional_edges("start", fan_out)

def process_item(state: dict) -> dict:
    return {"results": [state["item"].upper()]}

graph.add_node("process_item", process_item)
graph.add_edge("process_item", "aggregate")
```

---

## State Updates & Reducers Reference

```python
import operator
from langgraph.graph.message import add_messages

# Overwrite (default)
{"key": "new_value"}

# Append to list
Annotated[list, operator.add]

# Append messages (deduplicates by id)
Annotated[list[AnyMessage], add_messages]

# Custom reducer
def keep_last_5(existing: list, new: list) -> list:
    return (existing + new)[-5:]

Annotated[list, keep_last_5]
```

---

## Graph Visualization

```python
# ASCII
app.get_graph().print_ascii()

# PNG (requires Pillow + pygraphviz or mermaid)
img = app.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(img)

# Mermaid string
print(app.get_graph().draw_mermaid())
```

---

## LangSmith Tracing

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"]    = "ls__..."
os.environ["LANGCHAIN_PROJECT"]    = "langgraph-app"
# All graph runs are traced automatically
```

---

## Streaming Modes

```python
# "values"   — emit full state after each node
# "updates"  — emit only state changes after each node (default)
# "messages" — emit LLM tokens as they stream
# "debug"    — emit everything (verbose)

for chunk in app.stream(inputs, stream_mode="values"):
    print(chunk)

for chunk in app.stream(inputs, stream_mode="updates"):
    for node, update in chunk.items():
        print(f"{node}: {update}")

# Multiple modes at once
for mode, chunk in app.stream(inputs, stream_mode=["updates", "messages"]):
    print(mode, chunk)
```

---

## Config Schema (Runtime Parameters)

```python
from langgraph.graph import StateGraph
from pydantic import BaseModel

class Config(BaseModel):
    temperature: float = 0.7
    system_prompt: str = "You are helpful."

graph = StateGraph(State, config_schema=Config)

# Pass at runtime
config = {
    "configurable": {
        "thread_id": "t1",
        "temperature": 0.2,
        "system_prompt": "You are concise.",
    }
}
app.invoke(inputs, config=config)

# Access in nodes
def my_node(state, config):
    temp = config["configurable"].get("temperature", 0.7)
    ...
```

---

## Common Patterns

### ReAct (Reason + Act)
```
agent → tools_condition → tools → agent → ... → END
```

### RAG Agent
```
retrieve → grade_docs → generate → check_hallucination → END
                ↓ (no relevant docs)
            rewrite_query → retrieve
```

### Plan-and-Execute
```
planner → executor_loop → replanner → executor_loop → ... → END
```

### Reflection
```
generate → reflect → generate → reflect → ... → END (after N iterations)
```

---

## Tips & Gotchas

- Nodes should return **only the keys they modify** — unmentioned keys stay unchanged.
- Use `Annotated[list, operator.add]` for any state that multiple nodes write to.
- `MessagesState` + `add_messages` handles message deduplication by `id` automatically.
- Always set `thread_id` in config when using a checkpointer — without it, state is not saved.
- `interrupt_before` / `interrupt_after` require a checkpointer to work.
- `app.invoke(None, config=config)` resumes a paused graph from its last checkpoint.
- The `Send` API is the right tool for dynamic parallelism (e.g. map-reduce over a list).
- Subgraph state schemas don't need to match the parent — LangGraph handles the boundary.
- Use `stream_mode="messages"` for token-level streaming in chat UIs.
- Compile once, invoke many times — `graph.compile()` is not free, do it at startup.
