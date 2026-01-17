 The **MCP (Model Context Protocol) cheatsheet** is a quick reference for building and using MCP servers and clients. It covers installation, setup, server creation, tools/resources, and security considerations.  

---

## 📌 MCP Cheatsheet (Model Context Protocol)

### 🔹 Installation & Setup
- **Add MCP to project (pip)**  
  ```bash
  pip install "mcp[cli]"
  ```
- **Run dev tools**  
  ```bash
  mcp dev server.py
  ```

---

### 🔹 Quickstart Server Example
```python
# server.py
from mcp.server.fastmcp import FastMCP

# Create MCP server
mcp = FastMCP("Demo")

# Tool: Add two numbers
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

# Resource: Dynamic greeting
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}!"
```

Run with:
```bash
mcp dev server.py
```

---

### 🔹 Core Primitives
- **Tools** → Functions LLMs can invoke (actions).  
- **Resources** → File‑like data to read (context).  
- **Prompts** → Pre‑written message templates.  
- **Sampling** → Server requests LLM completion.  

---

### 🔹 Architecture
- **Host** → AI application (e.g., Claude Desktop, IDE).  
- **Client** → Protocol client inside host.  
- **Server** → Exposes tools/resources to clients.  
- **Transport** → `stdio` or HTTP with SSE.  
- **Integration** → One host connects to multiple servers.  

---

### 🔹 Security Considerations
- **Verify third‑party servers** before connecting.  
- **Follow OWASP MCP security guidelines**:
  - Robust client security.  
  - Secure server discovery.  
  - Strong governance for external tools.  

---

## 📚 Sources
- [MCP Cheat Sheet – Quick Reference](https://mcp-cheatsheet.online/)  
- [Tech Interview Prep – MCP Overview](https://taanqai.com/mcp)  
- [Securely Using Third‑Party MCP Servers](https://www.msbiro.net/posts/securely-using-third-party-mcp-servers/)

---

**visual flow diagram in Markdown text** that shows the **MCP (Model Context Protocol) architecture** from top to bottom:

```markdown
# 📊 MCP Architecture Flow

Start
  │
  ▼
[Host Application]
  │
  ▼
[Client (MCP Client inside Host)]
  │
  ▼
[Transport Layer]
  ├──► stdio
  └──► HTTP + SSE
  │
  ▼
[Server (MCP Server)]
  │
  ├──► Tools (functions/actions LLMs can invoke)
  │
  ├──► Resources (documents, APIs, file-like data)
  │
  ├──► Prompts (pre-written templates)
  │
  └──► Sampling (LLM completion requests)
  │
  ▼
[LLM / AI Agent]
  │
  ▼
[Response back to Host Application]
```

---

### 🔎 Explanation
- **Host Application** → The environment where MCP runs (e.g., Claude Desktop, IDE).  
- **Client** → The MCP client inside the host that communicates with servers.  
- **Transport Layer** → Defines how data flows (stdio or HTTP/SSE).  
- **Server** → Exposes tools, resources, prompts, and sampling to the client.  
- **LLM/Agent** → Uses these capabilities to generate context‑aware responses.  
- **Response** → Flows back to the host for user interaction.  

---

**handy MCP (Model Context Protocol) coding cheatsheet** so you can quickly recall the main primitives and decorators when building MCP servers:

---

## 📌 MCP Cheatsheet

### 🔹 Server Setup
```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("DemoServer")
```

---

### 🔹 Tools
- **Definition**: Functions that the LLM can call to perform actions.
```python
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b
```

---

### 🔹 Resources
- **Definition**: File‑like data sources that can be read by the client.
```python
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Return a personalized greeting"""
    return f"Hello, {name}!"
```

---

### 🔹 Prompts
- **Definition**: Pre‑written templates for consistent responses.
```python
@mcp.prompt("welcome")
def welcome_prompt(user: str) -> str:
    return f"Welcome, {user}! How can I help you today?"
```

---

### 🔹 Sampling
- **Definition**: Request the LLM to generate text from the server side.
```python
@mcp.sample()
def generate_story(topic: str) -> str:
    return f"Tell me a short story about {topic}."
```

---

### 🔹 Running the Server
```bash
mcp dev server.py
```

---

## ⚡ Quick Reference
- `@mcp.tool` → define callable functions.  
- `@mcp.resource` → expose data sources.  
- `@mcp.prompt` → reusable text templates.  
- `@mcp.sample` → request completions from LLM.  
- `FastMCP("Name")` → create server instance.  
- `mcp dev server.py` → run development server.  

---
