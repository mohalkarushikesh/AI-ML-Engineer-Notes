# LoRA Model Cheatsheet

> **LoRA (Low-Rank Adaptation)** — A parameter-efficient fine-tuning technique that freezes the original model weights and injects small trainable rank-decomposition matrices into each layer.

---

[Lora Blog](https://www.ibm.com/think/topics/lora)

---

## Core Concept

Instead of updating all model weights during fine-tuning:

```
W' = W + ΔW     where ΔW = A × B
```

- `W` — frozen pre-trained weight matrix (e.g. 4096×4096)
- `A` — down-projection matrix: `d × r`
- `B` — up-projection matrix: `r × d`
- `r` — **rank** (the key hyperparameter, typically 4–64)

Only **A** and **B** are trained. Parameters saved ≈ `2 × d × r` vs `d²`.

---

## Key Hyperparameters

| Parameter | Description | Typical Range | Notes |
|-----------|-------------|---------------|-------|
| `rank` (r) | Dimension of low-rank matrices | 4–128 | Higher = more expressive, more VRAM |
| `alpha` (α) | Scaling factor for LoRA output | 16–128 | Often set equal to rank; controls learning rate effectively |
| `dropout` | Dropout on LoRA layers | 0.0–0.1 | Regularization; 0 is common |
| `target_modules` | Which layers to apply LoRA | varies | See layer targets below |
| `bias` | Train bias terms | `"none"` / `"all"` / `"lora_only"` | Usually `"none"` |
| `task_type` | Model task type | `CAUSAL_LM`, `SEQ2SEQ`, etc. | Required for PEFT |

### Effective Scaling
```
output += (alpha / rank) × B × A × input
```
Setting `alpha = rank` gives a scale of `1.0` (no extra scaling).

---

## Rank Selection Guide

| Rank | Use Case | VRAM Impact |
|------|----------|-------------|
| 2–4 | Style transfer, small concept | Minimal |
| 8–16 | General fine-tuning, chat | Low |
| 32–64 | Complex task adaptation | Moderate |
| 128+ | Near full fine-tune quality | High |

> **Rule of thumb:** Start with `r=8`, `alpha=16`. Increase rank if the model underfits.

---

## Common Target Modules

### LLaMA / Mistral / Gemma
```python
target_modules = ["q_proj", "v_proj"]              # Minimal
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]  # Attention only
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]     # Full (recommended)
```

### GPT-2 / GPT-J
```python
target_modules = ["c_attn", "c_proj"]
```

### BERT / RoBERTa
```python
target_modules = ["query", "value"]
```

### Stable Diffusion (UNet)
```python
target_modules = ["to_q", "to_v", "to_k", "to_out.0"]
```

---

## Quick Setup (Hugging Face PEFT)

```python
from peft import LoraConfig, get_peft_model, TaskType

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(base_model, config)
model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.06%
```

---

## Merging LoRA Back into Base Model

```python
from peft import PeftModel

# Load and merge
model = PeftModel.from_pretrained(base_model, "path/to/lora/adapter")
merged_model = model.merge_and_unload()

# Save merged model
merged_model.save_pretrained("path/to/merged_model")
tokenizer.save_pretrained("path/to/merged_model")
```

---

## LoRA Variants

| Variant | Key Idea | Best For |
|---------|----------|----------|
| **LoRA** | Base rank decomposition | General fine-tuning |
| **QLoRA** | LoRA on 4-bit quantized model | Consumer GPU fine-tuning |
| **LoRA+** | Different LR for A and B matrices | Better convergence |
| **DoRA** | Decomposes weight into magnitude + direction | Better than LoRA at same rank |
| **LoftQ** | Quantization-aware LoRA init | QLoRA accuracy improvement |
| **rsLoRA** | Scales alpha by `√rank` instead of rank | Stable training at high ranks |
| **AdaLoRA** | Adaptive rank allocation per layer | Budget-constrained fine-tuning |
| **IA³** | Learned rescaling vectors | Fewer parameters than LoRA |

---

## QLoRA Setup (4-bit)

```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",          # NormalFloat4 — best for LLMs
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)

# Prepare for k-bit training
from peft import prepare_model_for_kbit_training
model = prepare_model_for_kbit_training(model)
```

---

## Training Configuration (SFTTrainer)

```python
from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,      # effective batch = 16
    learning_rate=2e-4,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    fp16=True,                          # or bf16=True for Ampere+
    logging_steps=10,
    save_strategy="epoch",
    max_seq_length=2048,
    dataset_text_field="text",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=lora_config,
)
trainer.train()
```

---

## VRAM Estimates (7B Model)

| Method | Precision | Approx VRAM |
|--------|-----------|-------------|
| Full fine-tune | BF16 | ~112 GB |
| LoRA | BF16 | ~60 GB |
| LoRA | FP16 | ~30 GB |
| QLoRA | INT8 + LoRA | ~16 GB |
| QLoRA | NF4 + LoRA | ~10 GB |

---

## Stable Diffusion LoRA (Diffusers)

```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# Load LoRA weights
pipe.load_lora_weights("path/to/lora", weight_name="adapter.safetensors")

# Control strength (0.0 = no effect, 1.0 = full)
pipe.fuse_lora(lora_scale=0.8)

image = pipe("your prompt", num_inference_steps=30).images[0]
```

### SD LoRA Naming Convention
| Filename suffix | Type |
|-----------------|------|
| `_lora.safetensors` | Standard LoRA |
| `_lycoris.safetensors` | LyCORIS (LoCon/LoHa) |
| `_locon.safetensors` | LoCon (conv layers included) |

---

## Inference with Adapter (PEFT)

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained("base_model_id")
model = PeftModel.from_pretrained(base_model, "adapter_path")
tokenizer = AutoTokenizer.from_pretrained("base_model_id")

inputs = tokenizer("Hello, how are you?", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Loading Multiple Adapters
```python
model.load_adapter("adapter_a", adapter_name="A")
model.load_adapter("adapter_b", adapter_name="B")

model.set_adapter("A")          # switch to adapter A
model.disable_adapter()         # use base model only

# Combine adapters (weighted average)
model.add_weighted_adapter(
    adapters=["A", "B"],
    weights=[0.7, 0.3],
    adapter_name="combined",
    combination_type="linear",
)
```

---

## Saving & Loading

```python
# Save only adapter weights (~MBs, not GBs)
model.save_pretrained("my_lora_adapter/")
# Saves: adapter_config.json + adapter_model.safetensors

# Load later
model = PeftModel.from_pretrained(base_model, "my_lora_adapter/")

# Push to Hub
model.push_to_hub("username/my-lora-adapter")
```

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Loss not decreasing | Rank too low / LR too small | Increase rank or `learning_rate` |
| Catastrophic forgetting | LR too high | Reduce LR; add more diverse data |
| OOM on consumer GPU | Model too large | Use QLoRA (4-bit) |
| Adapter not taking effect | Wrong `target_modules` | Check layer names with `model.named_modules()` |
| Slow convergence | alpha/rank ratio off | Try `alpha = 2 × rank` |
| NaN loss | BF16 instability | Switch to FP16 or add `gradient_clipping` |

---

## Useful Commands

```bash
# Inspect model layers
python -c "from transformers import AutoModel; m = AutoModel.from_pretrained('model_id'); print([n for n,_ in m.named_modules()])"

# Check trainable parameters
model.print_trainable_parameters()

# Convert .bin adapter to .safetensors
python -c "from safetensors.torch import save_file; import torch; save_file(torch.load('adapter_model.bin'), 'adapter_model.safetensors')"
```

---

## Resources

| Resource | URL |
|----------|-----|
| PEFT Docs | https://huggingface.co/docs/peft |
| LoRA Paper | https://arxiv.org/abs/2106.09685 |
| QLoRA Paper | https://arxiv.org/abs/2305.14314 |
| TRL (SFTTrainer) | https://huggingface.co/docs/trl |
| Diffusers LoRA | https://huggingface.co/docs/diffusers/training/lora |
| Axolotl (training framework) | https://github.com/OpenAccess-AI-Collective/axolotl |
