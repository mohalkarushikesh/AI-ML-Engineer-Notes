**Retrieval-Augmented Generation (RAG) cheatsheet** — a compact guide to the key concepts, workflows, and code snippets you’ll need when building RAG systems 🚀.  

---

## 🧩 Core Concepts
- **RAG = Retrieval + Generation**
  - Combines **LLMs** with **retrievers** (vector databases, search engines).
  - Ensures answers are grounded in external knowledge.
- **Retriever**: Finds relevant documents (FAISS, Pinecone, Weaviate, Chroma).  
- **Generator (LLM)**: Produces natural language output using retrieved context.  
- **Pipeline**: Query → Retrieve → Augment → Generate.  

---

## ⚙️ Workflow
1. **Embed documents** → Convert text into vectors.  
2. **Store in vector DB** → FAISS, Pinecone, Chroma, Weaviate.  
3. **Retrieve relevant docs** → Similarity search.  
4. **Augment prompt** → Add retrieved docs to LLM input.  
5. **Generate answer** → LLM produces grounded response.  

---

## 🔨 Code Snippets (Python + LangChain)

### 1. Setup
```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Embeddings + LLM
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-3.5-turbo")
```

### 2. Create Vector Store
```python
texts = ["LangChain is a framework", "RAG combines retrieval and generation"]
db = FAISS.from_texts(texts, embeddings)
retriever = db.as_retriever()
```

### 3. RetrievalQA Chain
```python
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"   # options: "stuff", "map_reduce", "refine"
)

response = qa.run("What is RAG?")
print(response)
```

---

## 📊 Common Chain Types
- **Stuff**: Concatenate retrieved docs → feed to LLM.  
- **Map-Reduce**: Summarize each doc → combine summaries.  
- **Refine**: Iteratively improve answer with each doc.  

---

## 🛠️ Best Practices
- Use **chunking** (e.g., 500–1000 tokens) for documents.  
- Apply **metadata filtering** (author, date, tags).  
- Cache embeddings for efficiency.  
- Evaluate with **precision/recall** on retrieval.  
- Add **citations** in generated answers.  

---

## 📚 Tools & Libraries
- **LangChain** → RAG pipelines.  
- **LlamaIndex** → Document indexing + retrieval.  
- **FAISS / Pinecone / Weaviate / Chroma** → Vector DBs.  
- **OpenAI / Anthropic / Cohere / HuggingFace** → LLMs.  

---
