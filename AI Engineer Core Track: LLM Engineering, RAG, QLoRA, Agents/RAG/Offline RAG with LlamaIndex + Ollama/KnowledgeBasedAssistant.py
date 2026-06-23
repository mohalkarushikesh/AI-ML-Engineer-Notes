# Knowledge Based Assistant

# pip install llama-index llama-index-llms-ollama llama-index-embeddings-ollama

import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# Initialize local Ollama models
Settings.llm = Ollama(model="qwen2.5:1.5b", request_timeout=120.0)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")  # ← change this

os.makedirs("data", exist_ok=True)
# Example: Create a dummy text file
with open("data/company_info.txt", "w") as f:
    f.write("LlamaIndex and Gemini API integration is seamless and fast.")

# Read your data folder
documents = SimpleDirectoryReader('data').load_data()

# Indexing
index = VectorStoreIndex.from_documents(documents)

# Ask questions
query_engine = index.as_query_engine()

response = query_engine.query('What is LlamaIndex?')
print(response)
