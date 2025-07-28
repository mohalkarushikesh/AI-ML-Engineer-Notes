## 🧠 What Is LangChain?

LangChain is an open-source framework that helps developers build applications powered by **Large Language Models (LLMs)**. It simplifies chaining together components like prompts, memory, tools, and external data sources to create intelligent workflows.

---

## 🔗 Integrations with LLMs

LangChain supports multiple LLM providers:
- **OpenAI** (GPT-3.5, GPT-4)
- **Anthropic** (Claude)
- **Google** (Gemini)
- **Cohere**, **DeepSeek**, **Mistral**, and more

You can switch between models easily using LangChain’s abstraction layers.

---

## ⚙️ How LangChain Works

LangChain connects components like:
- **Prompt templates** → structure model input
- **Chains** → define workflows
- **Agents** → make decisions and use tools
- **Memory** → retain context
- **Retrievers** → fetch relevant data
- **Tools** → external APIs or functions

These components are linked using **LangChain Expression Language (LCEL)** or **LangGraph** for orchestration.

---

## 📥 Importing Language Models

```python
from langchain.llms import OpenAI
llm = OpenAI(openai_api_key="your_key")
response = llm.predict("Hello!")
```

For chat models:
```python
from langchain.chat_models import ChatOpenAI
chat = ChatOpenAI(openai_api_key="your_key")
```

---

## 🧾 Prompt Templates

Prompt templates help format inputs dynamically:
```python
from langchain.prompts import PromptTemplate
template = PromptTemplate.from_template("Translate {text} to French.")
prompt = template.format(text="Hello")
```

Supports few-shot examples, semantic selectors, and chaining.

---

## 🔗 Chains

Chains are sequences of components:
- **Simple chains**: One LLM call
- **Multi-step chains**: Combine prompts, tools, memory, etc.

Example:
```python
chain = prompt | llm
```

---

## 🗂️ Indexes

Used for **Retrieval-Augmented Generation (RAG)**.

### 📄 Document Loaders
Load data from:
- PDFs, CSVs, websites, Notion, Google Docs, etc.

### 🧠 Vector Databases
Store embeddings for semantic search:
- Chroma, Pinecone, FAISS, Weaviate

### ✂️ Text Splitters
Split documents into chunks for embedding:
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
```

### 🔍 Retrieval
Fetch relevant chunks using similarity search:
```python
retriever = vectorstore.as_retriever()
```

---

## 🧠 Memory

Memory stores context across interactions:
- **ConversationBufferMemory**
- **EntityMemory**
- **SummaryMemory**

Useful for chatbots and agents.

---

## 🛠️ Tools

Tools are external functions agents can use:
- Calculator
- Web search
- APIs
- Custom Python functions

---

## 🕵️ LangChain Agents

Agents decide what tools to use based on input:
- Use LLMs to reason and act
- Can call multiple tools
- Support function calling and toolkits

---

## 🔄 LangGraph

LangGraph is a stateful orchestration framework:
- Build multi-step, multi-actor workflows
- Supports streaming, persistence, and branching logic

---

## 🧪 LangSmith

LangSmith is a developer platform for:
- Debugging chains and agents
- Monitoring performance
- Evaluating outputs
- Logging traces

---

## 🚀 Getting Started

### 📦 Installation
```bash
pip install langchain openai python-dotenv
```

### 🧪 Setup
Create `.env` file:
```
OPENAI_API_KEY='your_key'
```

### 🧰 Use Cases
- **Chatbots**: Context-aware conversations
- **Summarization**: Condense long documents
- **Question Answering**: Pull answers from data
- **Data Augmentation**: Enrich datasets
- **Virtual Agents**: Autonomous decision-makers

---

## 📚 Resources

- [LangChain Official Docs](https://python.langchain.com/docs/introduction/)
- [LangChain Full Course on GitHub](https://github.com/Coding-Crashkurse/Langchain-Full-Course)
- [GeeksforGeeks LangChain Guide](https://www.geeksforgeeks.org/artificial-intelligence/introduction-to-langchain/)
- [LangChain Tutorial by Nanonets](https://nanonets.com/blog/langchain/)

---
