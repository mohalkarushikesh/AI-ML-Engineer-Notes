# LangChain Cheatsheet

> Build LLM-powered applications with chains, agents, retrievers, and more.

---

## Installation

```bash
pip install langchain langchain-core langchain-community
pip install langchain-openai      # OpenAI models
pip install langchain-anthropic   # Anthropic models
pip install langchain-google-genai # Google models
```

---

## Chat Models

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key="sk-...")
llm = ChatAnthropic(model="claude-opus-4-6", temperature=0)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

# Invoke
response = llm.invoke("What is the capital of France?")
print(response.content)

# Stream
for chunk in llm.stream("Tell me a joke"):
    print(chunk.content, end="", flush=True)

# Batch
responses = llm.batch(["Question 1", "Question 2"])
```

---

## Messages

```python
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Hello!"),
    AIMessage(content="Hi! How can I help?"),
    HumanMessage(content="Tell me about LangChain."),
]

response = llm.invoke(messages)
```

---

## Prompt Templates

```python
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    MessagesPlaceholder,
)

# Simple string template
prompt = PromptTemplate.from_template("Tell me a joke about {topic}")
prompt.format(topic="cats")

# Chat prompt
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in {domain}."),
    ("human", "{question}"),
])
chat_prompt.format_messages(domain="Python", question="What are decorators?")

# With message history placeholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])

# Few-shot
from langchain_core.prompts import FewShotChatMessagePromptTemplate

examples = [
    {"input": "2+2", "output": "4"},
    {"input": "3+3", "output": "6"},
]
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"), ("ai", "{output}")
])
few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
)
```

---

## LCEL — LangChain Expression Language

```python
# Basic chain: prompt | llm | parser
from langchain_core.output_parsers import StrOutputParser

chain = prompt | llm | StrOutputParser()
result = chain.invoke({"topic": "cats"})

# Parallel execution
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

chain = RunnableParallel(
    joke=prompt_joke | llm | StrOutputParser(),
    fact=prompt_fact | llm | StrOutputParser(),
)

# Pass input through unchanged
chain = RunnablePassthrough() | llm

# Assign new keys
from langchain_core.runnables import RunnablePassthrough
chain = RunnablePassthrough.assign(summary=summary_chain)

# Lambda / custom function
from langchain_core.runnables import RunnableLambda

chain = RunnableLambda(lambda x: x["text"].upper())

# Conditional branching
from langchain_core.runnables import RunnableBranch

branch = RunnableBranch(
    (lambda x: "python" in x["topic"], python_chain),
    (lambda x: "java" in x["topic"], java_chain),
    default_chain,
)
```

---

## Output Parsers

```python
from langchain_core.output_parsers import (
    StrOutputParser,       # raw string
    JsonOutputParser,      # parse JSON
    CommaSeparatedListOutputParser,
)

# Pydantic structured output
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class Joke(BaseModel):
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline")

parser = PydanticOutputParser(pydantic_object=Joke)

prompt = PromptTemplate(
    template="Tell a joke.\n{format_instructions}\n",
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
chain = prompt | llm | parser

# .with_structured_output() (preferred for newer models)
llm_structured = llm.with_structured_output(Joke)
joke = llm_structured.invoke("Tell me a joke")
joke.setup      # typed attribute
joke.punchline
```

---

## Document Loaders

```python
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    WebBaseLoader,
    DirectoryLoader,
    UnstructuredMarkdownLoader,
    JSONLoader,
)

# Text
loader = TextLoader("file.txt")
docs = loader.load()

# PDF
loader = PyPDFLoader("document.pdf")
pages = loader.load_and_split()

# Web
loader = WebBaseLoader("https://example.com")
docs = loader.load()

# Directory (all .txt files)
loader = DirectoryLoader("./docs", glob="**/*.txt", loader_cls=TextLoader)
docs = loader.load()

# CSV
loader = CSVLoader("data.csv", source_column="url")
docs = loader.load()
```

---

## Text Splitters

```python
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
)

# Recursive (recommended default)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""],
)
chunks = splitter.split_documents(docs)
chunks = splitter.create_documents(["raw text..."])

# Token-based
splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=50)

# Markdown by headers
headers = [("#", "H1"), ("##", "H2"), ("###", "H3")]
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
chunks = splitter.split_text(markdown_text)
```

---

## Embeddings

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Embed text
vector = embeddings.embed_query("Hello world")
vectors = embeddings.embed_documents(["doc 1", "doc 2"])
```

---

## Vector Stores

```python
# Chroma
from langchain_chroma import Chroma
vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="./db")
vectorstore = Chroma(persist_directory="./db", embedding_function=embeddings)

# FAISS
from langchain_community.vectorstores import FAISS
vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local("faiss_index")
vectorstore = FAISS.load_local("faiss_index", embeddings)

# Pinecone
from langchain_pinecone import PineconeVectorStore
vectorstore = PineconeVectorStore.from_documents(docs, embeddings, index_name="my-index")

# Common operations
vectorstore.add_documents(new_docs)
results = vectorstore.similarity_search("query", k=4)
results = vectorstore.similarity_search_with_score("query", k=4)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
```

---

## Retrievers

```python
# Basic vector retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",          # "similarity" | "mmr" | "similarity_score_threshold"
    search_kwargs={"k": 5},
)

# MMR (Maximal Marginal Relevance — reduces redundancy)
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5},
)

# Score threshold
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.7},
)

# Multi-query retriever
from langchain.retrievers import MultiQueryRetriever
retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)

# Contextual compression
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever,
)

# Ensemble (combine multiple retrievers)
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

bm25 = BM25Retriever.from_documents(docs)
ensemble = EnsembleRetriever(
    retrievers=[bm25, vector_retriever],
    weights=[0.5, 0.5],
)

# Invoke
docs = retriever.invoke("search query")
```

---

## RAG Chain

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context below.

Context: {context}

Question: {question}
""")

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("What is LangChain?")
```

---

## Memory / Chat History

```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])

chain = prompt | llm | StrOutputParser()

with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

with_history.invoke(
    {"input": "Hello!"},
    config={"configurable": {"session_id": "user_123"}},
)
```

---

## Tools

```python
from langchain_core.tools import tool
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun

# Custom tool with decorator
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

# Custom tool with schema
from pydantic import BaseModel
class SearchInput(BaseModel):
    query: str

@tool(args_schema=SearchInput)
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

# Built-in tools
search = DuckDuckGoSearchRun()
wiki = WikipediaQueryRun()

# Bind tools to LLM
llm_with_tools = llm.bind_tools([multiply, search])
```

---

## Agents

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor

tools = [search, multiply]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("agent_scratchpad"),
    ("human", "{input}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)

executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=10,
    handle_parsing_errors=True,
)

result = executor.invoke({"input": "What is 42 multiplied by 7?"})
print(result["output"])
```

---

## LangGraph (Stateful Agents)

```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from typing import TypedDict, Annotated
import operator

# Quick ReAct agent
agent = create_react_agent(llm, tools)
result = agent.invoke({"messages": [("human", "Search for LangChain news")]})

# Custom graph
class State(TypedDict):
    messages: Annotated[list, operator.add]

graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"continue": "tools", "end": END})
graph.add_edge("tools", "agent")
app = graph.compile()
```

---

## Callbacks & Tracing

```python
from langchain_core.callbacks import StdOutCallbackHandler
from langchain.callbacks import LangChainTracer   # LangSmith

# Verbose logging
chain.invoke({"input": "hello"}, config={"callbacks": [StdOutCallbackHandler()]})

# LangSmith tracing
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls__..."
os.environ["LANGCHAIN_PROJECT"] = "my-project"
# All chain calls are now traced automatically
```

---

## Caching

```python
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache, SQLiteCache

# In-memory cache
set_llm_cache(InMemoryCache())

# SQLite persistent cache
set_llm_cache(SQLiteCache(database_path=".langchain.db"))
```

---

## Streaming

```python
# Sync stream
for chunk in chain.stream({"topic": "space"}):
    print(chunk, end="", flush=True)

# Async stream
async for chunk in chain.astream({"topic": "space"}):
    print(chunk, end="", flush=True)

# Stream events (detailed)
async for event in chain.astream_events({"topic": "space"}, version="v2"):
    if event["event"] == "on_chat_model_stream":
        print(event["data"]["chunk"].content, end="")
```

---

## Async

```python
import asyncio

# All runnables support async
result = await chain.ainvoke({"input": "hello"})
results = await chain.abatch([{"input": "q1"}, {"input": "q2"}])
docs = await retriever.ainvoke("query")
```

---

## Runnable Config

```python
# Tags, metadata, callbacks, max concurrency
chain.invoke(
    {"input": "hello"},
    config={
        "tags": ["production"],
        "metadata": {"user_id": "u123"},
        "max_concurrency": 5,
        "run_name": "my_run",
    }
)
```

---

## Inspecting Chains

```python
# View input/output schema
chain.input_schema.schema()
chain.output_schema.schema()

# Visualize (requires graphviz)
chain.get_graph().print_ascii()

# Get intermediate steps
chain.with_config({"return_intermediate_steps": True})
```

---

## Key Packages

| Package | Purpose |
|---------|---------|
| `langchain-core` | Base abstractions, LCEL |
| `langchain` | Chains, agents, memory |
| `langchain-community` | 3rd-party integrations |
| `langchain-openai` | OpenAI / Azure OpenAI |
| `langchain-anthropic` | Anthropic Claude |
| `langchain-google-genai` | Google Gemini |
| `langchain-chroma` | Chroma vector store |
| `langchain-pinecone` | Pinecone vector store |
| `langchain-text-splitters` | Text chunking |
| `langgraph` | Stateful multi-agent graphs |
| `langsmith` | Tracing & evaluation |

---

## Tips & Gotchas

- Prefer **LCEL** (`|` operator) over legacy `LLMChain` / `ConversationalRetrievalChain`.
- Use `.with_structured_output()` over `PydanticOutputParser` when the model supports it.
- `RunnablePassthrough()` is your friend for passing raw input alongside processed data.
- `batch()` is faster than looping `invoke()` — it runs calls concurrently.
- Always set `temperature=0` for deterministic/factual tasks.
- Use `max_iterations` in `AgentExecutor` to prevent infinite loops.
- Enable LangSmith tracing early — it makes debugging chains much easier.
- `langchain-community` requires separate installs for many integrations (e.g. `pip install faiss-cpu`).
