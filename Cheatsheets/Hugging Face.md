Here’s a **clean, no-BS Hugging Face cheat sheet** — focused on **daily usage, interviews, and real projects** (RAG, LLMs, fine-tuning).

---

# 🤗 Hugging Face Cheat Sheet

---

## 🔹 What Hugging Face Is

Hugging Face = **GitHub for AI**

* Models
* Datasets
* Tokenizers
* Inference APIs
* Spaces (apps)
* Training & fine-tuning tools

---

## 🔹 Install Core Libraries

```bash
pip install transformers datasets accelerate tokenizers sentencepiece
```

Optional (for fine-tuning):

```bash
pip install peft bitsandbytes trl
```

---

## 🔹 Login (Important)

```bash
huggingface-cli login
```

Token from → hf.co/settings/tokens

---

## 🔹 Load Model & Tokenizer

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "meta-llama/Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
```

---

## 🔹 Text Generation (Basic)

```python
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="gpt2"
)

pipe("Explain transformers in simple terms", max_new_tokens=100)
```

---

## 🔹 Chat Models (Instruct)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

inputs = tokenizer("Explain RAG", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

---

## 🔹 Quantization (Low RAM 🔥)

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_4bit=True)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
```

👉 Works well on **8–16GB RAM**

---

## 🔹 Sentence Embeddings

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
emb = model.encode("Hugging Face is awesome")
```

Used in:

* RAG
* Search
* Recommendation
* Clustering

---

## 🔹 Load Dataset

```python
from datasets import load_dataset

dataset = load_dataset("squad")
print(dataset["train"][0])
```

---

## 🔹 Custom Dataset from CSV

```python
dataset = load_dataset("csv", data_files="data.csv")
```

---

## 🔹 Fine-Tuning (Trainer API)

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./model",
    per_device_train_batch_size=2,
    num_train_epochs=3
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"]
)

trainer.train()
```

---

## 🔹 PEFT / LoRA (Must Know 💥)

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05
)

model = get_peft_model(model, config)
```

✅ Cheap fine-tuning
✅ Industry standard

---

## 🔹 Push Model to Hub

```python
model.push_to_hub("username/my-model")
tokenizer.push_to_hub("username/my-model")
```

---

## 🔹 Hugging Face Inference API

```python
import requests

API_URL = "https://api-inference.huggingface.co/models/gpt2"
headers = {"Authorization": "Bearer HF_TOKEN"}

response = requests.post(
    API_URL,
    headers=headers,
    json={"inputs": "Explain LLMs"}
)

print(response.json())
```

---

## 🔹 Spaces (Apps)

Frameworks supported:

* Gradio
* Streamlit
* Static HTML

Example:

```python
import gradio as gr

def chat(msg):
    return msg.upper()

gr.Interface(chat, "text", "text").launch()
```

---

## 🔹 Hugging Face CLI

```bash
huggingface-cli repo create my-model
huggingface-cli whoami
huggingface-cli download gpt2
```

---

## 🔹 Model Types (Interview Gold)

| Task      | Pipeline                       |
| --------- | ------------------------------ |
| Text Gen  | `text-generation`              |
| Chat      | `chat-completion`              |
| Embedding | `feature-extraction`           |
| Image     | `image-classification`         |
| Speech    | `automatic-speech-recognition` |

---

## 🔹 Best Models (Low Resource)

* `mistral-7b-instruct`
* `phi-2`
* `phi-3-mini`
* `all-MiniLM-L6-v2` (embeddings)
* `deepseek-coder-6.7b`

---

## 🔹 Hugging Face vs Ollama

| Hugging Face   | Ollama           |
| -------------- | ---------------- |
| Cloud + Local  | Local only       |
| Training + FT  | Inference        |
| Huge ecosystem | Simple           |
| Best for R&D   | Best for offline |

---

## 🔥 Real-World Uses

* RAG systems
* Fine-tuned chatbots
* Code assistants
* AI search engines
* Enterprise LLM pipelines

---

## 🧠 One-Line Summary

**Hugging Face is the backbone of modern AI development — models, data, training, and deployment in one ecosystem.**

---

If you want next:

* **Hugging Face + RAG full pipeline**
* **HF vs OpenAI vs Ollama architecture**
* **Interview Q&A on Hugging Face**
* **Offline HF workflow (no internet)**

Say the word 🚀
