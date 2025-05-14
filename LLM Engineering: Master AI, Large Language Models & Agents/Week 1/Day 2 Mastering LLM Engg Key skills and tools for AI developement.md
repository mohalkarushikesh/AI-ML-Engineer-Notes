## **Frontier Models Overview**  
### **Closed-Source Frontier Models**  
- **GPT** – OpenAI  
- **Claude** – Anthropic  
- **Gemini** – Google  
- **Command R** – Cohere  
- **Perplexity** – Perplexity AI  

### **Open-Source Frontier Models**  
- **Llama** – Meta  
- **Mistral** – Mistral AI  
- **Qwen** – Alibaba Cloud  
- **Gemma** – Google  
- **Phi** – Microsoft  

---

## **Three Ways to Use Large Language Models (LLMs)**  
1️⃣ **Chat Interface**  
   - Example: ChatGPT  

2️⃣ **Cloud API (LLM APIs)**  
   - Frameworks: LangChain  
   - Managed AI Cloud Services:  
     - Amazon Bedrock  
     - Google Vertex AI  
     - Azure ML  

3️⃣ **Direct Interface (Running Locally)**  
   - Via Hugging Face **Transformers** library  
   - Using **Ollama**  

---

## **How to Use Ollama with a Local LLM Interface**  
### **Setup Steps**  
1. **Install Ollama** and start it  
   - Check **localhost:11424**  
   - If needed, run: `ollama server`  
2. **Import necessary libraries**  
3. **Set up constants**  

### **Demo Code for a Question-Answering Pipeline**  
```python
from transformers import pipeline

# Load the QA pipeline
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Define the context and question
context = "The Transformer model was introduced in a paper called 'Attention Is All You Need' by Vaswani et al. in 2017."
question = "Who introduced the Transformer model?"

# Get the answer
answer = qa_pipeline(question=question, context=context)
print("Answer:", answer["answer"])
```

---

## **LLM Capabilities**  
✔️ **Question Answering (QA)**  
✔️ **Text Summarization**  
✔️ **Code Generation**  
✔️ **Reasoning & Logical Analysis**  
✔️ **Classification**  

---

## **Prompting Techniques**  
🔹 **Zero-shot prompting**  
🔹 **Few-shot prompting**  
🔹 **Chain of Thought (CoT)**  
🔹 **Role-based prompting** – Behave as an **engineer**, **doctor**, etc.  
🔹 **Prompt chaining**  

---

## **LangChain Framework**  
LangChain is a framework that simplifies the process of building applications powered by large language models (LLMs).  

---

## **Toolchains & Orchestration**  
🛠 **Retrieval-Augmented Generation (RAG)**  
🛠 **Microsoft 365 Copilot**  
🛠 **Agentic AI**  

---

## **Performance of Frontier Models**  
✔️ **Synthesizing Information**  
✔️ **Expanding on Skeleton Ideas**  
✔️ **Coding & Development**  

### **Limitations**  
⚠️ **Struggles in Specialized Domains**  
⚠️ **Lacks Knowledge of Recent Events**  
⚠️ **Can Make Confident Mistakes**  

---

 
