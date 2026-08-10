1. **AI Agents : Models that don't just answer questions but can:**

> Plan tasks
> Use tools
> Browse applications
> Execute workflows autonomously



Here's a spread of genuinely advanced AI/ML projects, organized by what skill they force you to develop. "Advanced" here means projects where the hard part isn't calling an API but engineering the system around it.

**LLM / NLP systems**

Build a retrieval-augmented generation system from scratch that actually handles the hard parts: chunking strategy, hybrid search (dense + BM25), reranking, and handling contradictory sources. The naive version is a weekend tutorial; the advanced version is making retrieval quality measurable and tuning it. Pair it with an eval harness so you can prove improvements rather than vibe-check them.

**Fine-tune a small open model (Llama, Qwen, Mistral) with LoRA/QLoRA on a domain-specific task**, then quantize it and deploy it to run on constrained hardware. The learning is in the full loop: dataset curation, training instability, evaluation, and the accuracy-vs-latency tradeoffs of quantization.

Build an agent that uses tools and can recover from its own mistakes. Multi-step reasoning, tool calling, error handling, and guardrails against loops. The interesting engineering is in observability and making failures debuggable.

**Computer vision**

Train a model for a task with genuinely scarce or imbalanced data, forcing you into augmentation, self-supervised pretraining, or synthetic data generation. Real-world CV is rarely about ImageNet-clean datasets.

Build a real-time inference pipeline (object detection or segmentation) optimized for edge deployment — model distillation, ONNX/TensorRT conversion, and hitting a latency budget.

**Reinforcement learning**

Implement PPO or a similar algorithm from scratch and train it on a non-trivial environment. RL is where "it should work" and "it works" diverge most; the debugging skills transfer everywhere.

RLHF-style preference tuning on a small model — collect preference data, train a reward model, run the policy optimization. This mirrors how frontier models are actually aligned.

**MLOps / systems**

Build an end-to-end pipeline with data versioning, experiment tracking, automated retraining triggered by data drift, and monitoring in production. This is what separates "I trained a model" from "I run models." Often the most employable skill and the most neglected in portfolios.

**From-scratch implementations**

Implement a transformer (attention, positional encoding, the training loop) with no high-level libraries, then train it on a small corpus. Doing this once changes how you read papers permanently.

---

A few of these would benefit from knowing more about your goal — a research portfolio, a job-hunting portfolio, and a "learn the internals deeply" path point at pretty different choices. If you tell me your target (domain, whether you have GPU access, and how much time), I can narrow this to two or three and sketch a concrete build plan with datasets and milestones. Want me to do that?
