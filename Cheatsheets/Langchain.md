**LangChain cheatsheet** highlighting the most important concepts, commands, and patterns you’ll use when building LLM-powered applications.  

---

## ⚙️ Setup
```bash
# Installation
pip install langchain
pip install langchain-openai
pip install langchain-community
pip install langchain-anthropic
```

```python
# Basic imports
from langchain_openai import OpenAI, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
```

---

## 🧩 Core Concepts
- **LLMs**: Large language models (e.g., OpenAI, Anthropic, Cohere).  
- **Chains**: Sequences of calls combining models, prompts, and logic.  
- **Prompts**: Templates that structure input for LLMs.  
- **Memory**: Store conversation history or context.  
- **Agents**: LLMs that decide which tools/actions to use.  
- **Retrieval Augmented Generation (RAG)**: Connect LLMs to external data sources.  
- **LCEL (LangChain Expression Language)**: Declarative way to compose chains.  

---

## 💬 Models & Prompts
```python
# LLM
llm = OpenAI(model_name="text-davinci-003", temperature=0.7)

# Chat Model
chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

# Prompt Template
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in simple terms."
)
```

---

## 🔗 Chains
```python
chain = LLMChain(llm=llm, prompt=prompt)
response = chain.run("LangChain")
print(response)
```

---

## 🧠 Memory
```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
chain = LLMChain(llm=llm, prompt=prompt, memory=memory)
```

---

## 🤖 Agents
```python
from langchain.agents import initialize_agent, Tool

tools = [
    Tool(name="Search", func=lambda q: "Result for " + q, description="Search tool")
]

agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
agent.run("Find LangChain cheatsheet")
```

---

## 📂 Document Processing (RAG)
```python
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Split text
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_text("Your document text here")

# Embeddings + Vector Store
embeddings = OpenAIEmbeddings()
db = FAISS.from_texts(docs, embeddings)

# Retrieval
retriever = db.as_retriever()
```

---

## 🚀 Key Patterns
- **LLMChain** → simplest building block.  
- **SequentialChain** → run multiple chains in order.  
- **RouterChain** → route inputs to different chains.  
- **AgentExecutor** → let LLMs decide which tool to call.  

---

Sources: You can explore detailed references at [Developer Updates LangChain Cheatsheet](https://www.developerupdates.com/cheatsheets/langchain) and [Cheat Sheets Hero LangChain Python Cheatsheet](https://cheatsheetshero.com/user/all/946-langchain-python-cheatsheet).  

---
