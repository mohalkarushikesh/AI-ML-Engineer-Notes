Here’s a **clean, practical Ollama cheat sheet** you can use daily 👇
(works well even with limited system access, and fits your AI/LLM learning path)

---

## 🦙 Ollama Cheat Sheet

### 🔹 Install & Verify



```bash
Step 1: Install Ollama via WSL 
curl -fsSL https://ollama.com/install.sh | sh
   - Debian/Ubuntu: sudo apt-get install zstd

Step 2: Start the Service
  sudo systemctl start ollama

Step 3: Run a Model
ollama run llama3

ollama --version
ollama list
```

---

### 🔹 Run a Model (Basic)

```bash
ollama run llama3
```

Interactive chat starts immediately.

Run with prompt:

```bash
ollama run llama3 "Explain transformers in simple terms"
```

---

### 🔹 Popular Models

```bash
ollama run llama3
ollama run mistral
ollama run phi
ollama run gemma
ollama run codellama
ollama run deepseek-coder
```

List all installed models:

```bash
ollama list
```

---

### 🔹 Download (Pull) Models

```bash
ollama pull llama3
ollama pull mistral:7b
ollama pull codellama:13b
```

Remove model:

```bash
ollama rm llama3
```

---

### 🔹 Model Tags & Sizes

```bash
ollama pull llama3:8b
ollama pull llama3:70b
```

Check details:

```bash
ollama show llama3
```

---

### 🔹 Create Custom Model (Modelfile)

Create file:

```bash
nano Modelfile
```

Example:

```text
FROM llama3
SYSTEM You are an expert Java backend engineer.
PARAMETER temperature 0.3
```

Build:

```bash
ollama create java-ai -f Modelfile
```

Run:

```bash
ollama run java-ai
```

---

### 🔹 Serve Ollama as API

Start server:

```bash
ollama serve
```

Default endpoint:

```
http://localhost:11434
```

---

### 🔹 API Call (cURL)

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "llama3",
  "prompt": "Explain RAG in 3 lines"
}'
```

---

### 🔹 Python Usage

```python
import requests

url = "http://localhost:11434/api/generate"
data = {
    "model": "llama3",
    "prompt": "Explain attention mechanism"
}

response = requests.post(url, json=data, stream=True)
for line in response.iter_lines():
    if line:
        print(line.decode())
```

---

### 🔹 Embeddings

```bash
curl http://localhost:11434/api/embeddings -d '{
  "model": "llama3",
  "prompt": "Ollama embeddings"
}'
```

---

### 🔹 Environment Variables

```bash
export OLLAMA_HOST=0.0.0.0
export OLLAMA_MODELS=/custom/path
```

---

### 🔹 Performance Tips (Low RAM)

```bash
ollama run phi
ollama run mistral:7b
```

Best for **8–16 GB RAM**:

* `phi`
* `mistral`
* `llama3:8b`
* `deepseek-coder:6.7b`

---

### 🔹 Stop Running Model

```bash
Ctrl + C
```

---

### 🔹 Logs & Debug

```bash
ollama logs
```

---

### 🔹 Update Ollama

```bash
ollama update
```

---

## ⚡ Best Use Cases

* Local ChatGPT replacement
* RAG systems
* Code assistant (Java, backend, system design)
* Offline AI learning
* Secure enterprise environments

---

If you want, I can also give you:

* **Ollama + LangChain cheat sheet**
* **Ollama + Spring Boot (Java) integration**
* **Best Ollama models for 8GB RAM**
* **Offline RAG setup using Ollama**

Just say the word 🔥
