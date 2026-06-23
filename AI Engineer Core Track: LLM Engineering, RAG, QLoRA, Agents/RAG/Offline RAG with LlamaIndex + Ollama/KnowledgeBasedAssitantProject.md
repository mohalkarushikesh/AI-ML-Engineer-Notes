## Level 2 — Use Your Own Real Documents
Instead of a hardcoded string, load actual files:
```python
# Drop PDFs, txt, docx files into data/ and just do:
documents = SimpleDirectoryReader('data').load_data()
```

## Level 3 — Persist the Index (Don't Re-embed Every Run)
```python
from llama_index.core import StorageContext, load_index_from_storage

PERSIST_DIR = "./storage"

if os.path.exists(PERSIST_DIR):
    # Load existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
else:
    # Build and save index
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
```

## Level 4 — Add Chat Memory (Multi-turn Conversation)
```python
chat_engine = index.as_chat_engine(chat_mode="condense_question")

while True:
    user_input = input("You: ")
    if user_input == "exit":
        break
    response = chat_engine.chat(user_input)
    print(f"Bot: {response}")
```

## Level 5 — Add a UI with Streamlit
```python
# pip install streamlit
import streamlit as st

st.title("Local RAG Chatbot")
query = st.text_input("Ask a question:")
if query:
    response = query_engine.query(query)
    st.write(response)
```

---

## Suggested path

```
Level 2 (real docs) → Level 3 (persist) → Level 4 (chat) → Level 5 (UI)
```

