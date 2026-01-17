**LangGraph** is a framework for building **stateful, graph-based AI agents**. A cheatsheet helps you quickly recall its core concepts: graphs, nodes, edges, state management, and orchestration patterns. You can find a detailed LangGraph cheatsheet [here](https://sumanmichael.github.io/langgraph-cheatsheet/).

---

## 🧩 Core Concepts
- **Graph**: Blueprint of your workflow. Nodes = components, Edges = connections.  
- **StateGraph**: General-purpose graph for managing state across workflows.  
- **MessageGraph**: Specialized for chatbots and conversational agents.  
- **Nodes**: Units of computation (LLM calls, tools, memory updates).  
- **Edges**: Define control flow between nodes.  
- **Checkpoints**: Save agent state (SQLite/Postgres integrations).  
- **Control Flow**: Branching, looping, conditional execution.  

---

## ⚙️ Installation
```bash
# Python
pip install langgraph langchain_openai
pip install langgraph langchain_anthropic
pip install langgraph-checkpoint-sqlite   # local persistence
pip install langgraph-checkpoint-postgres # production persistence
```

---

## 🔨 Basic Usage
```python
from langgraph.graph import StateGraph

# Define a simple graph
graph = StateGraph()

# Add nodes
graph.add_node("start", lambda state: {"msg": "Hello"})
graph.add_node("process", lambda state: {"msg": state["msg"] + " World"})

# Add edges
graph.add_edge("start", "process")

# Compile and run
app = graph.compile()
result = app.invoke({})
print(result)  # {'msg': 'Hello World'}
```

---

## 🧠 Memory & State
- **State**: Dict-like object passed between nodes.  
- **Persistence**: Use checkpoints to save/restore state.  
- **Long-running agents**: State survives across multiple invocations.  

---

## Advanced Concepts

- **Cyclic Graphs**: Agents repeatedly call tools and refine information (e.g., research) until a condition is met, unlike simple DAGs.
- **Orchestrator-Worker Models**: An orchestrator sends tasks (e.g., sections of a document) to specialized workers, collects their outputs in shared state, and synthesizes a final result.
- **Agent Swarms**: Multiple agents interact, handing off conversations based on intent (e.g., flight assistant to hotel assistant).
- **State Management**: Graphs use a shared State (like MessagesState) that nodes read from and write to, enabling memory and complex interactions. 

---

## 🤖 Agent Patterns
- **Single Agent**: One graph controlling all logic.  
- **Multi-Agent**: Multiple graphs interacting.  
- **Hierarchical**: Supervisor agent delegates to sub-agents.  
- **Human-in-the-loop**: Pause execution for user feedback.  

---

## 🛠️ Best Practices
- Keep nodes small and modular.  
- Use **edges** to define clear control flow.  
- Persist state for reliability.  
- Debug with logging at node boundaries.  
- Start simple (StateGraph) → scale to complex (MessageGraph).  

---

## When to Use LangGraph
- When your AI needs to remember things over time (**stateful**).
- When the flow isn't just forward-only (**needs loops/cycles**).
- When multiple AI components **need to collaborate**.
- When you need **fine-grained control** over LLM decision-making. 

---

## 📚 Resources
- [LangGraph Cheatsheet](https://sumanmichael.github.io/langgraph-cheatsheet/)  
- [TLDR LangGraph Development Guide](https://sumanmichael.github.io/langgraph-cheatsheet/cheatsheet/tldr/)  
- [Getting Started Guide](https://sumanmichael.github.io/langgraph-cheatsheet/cheatsheet/getting-started/)  

---
