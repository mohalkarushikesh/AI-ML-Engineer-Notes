Here's the library. **But read this first:** this is a reference shelf to pull from once you have a real interview date — not a to-do list for this week. This week is still just the conversation plus two evenings on your own repo.

## 0. Start here (free, and highest value for you)

Your own repo. Genuinely. The signal interviewers want is specific: whether you know what a token costs at scale, what an eval harness actually looks like, why your RAG retrieval is failing, and how to debug a multi-agent loop that silently drifted off plan. You have a system that touches most of that.

## 1. AI Engineer question banks

- **DataCamp** — 30 RAG interview questions, foundational to advanced. Also has separate sets for LLMs, generative AI, and agentic AI.
- **InterviewBit LLM guide** — transformers, LoRA/QLoRA/PEFT, RAG pipelines, tokenization, embeddings, hallucination mitigation, LLM system design, deployment.
- **Let's Data Science** — 50 questions across transformer fundamentals through safety/alignment, sourced from real 2026 interview loops.
- **ayautomate** — 40 questions with strong/weak sample answers, a scoring rubric, and take-home ideas. The rubric is the useful part; you can self-grade.
- **CallSphere** — 8 hard questions from real loops at frontier labs, with what the interviewer is actually testing. Note their most-asked 2026 question: **RAG vs fine-tuning vs both** — and they want a decision framework, not a definition. Have an answer ready.

## 2. Evaluation (your biggest gap — prioritise this)

- **Ragas** — the RAG Triad: Faithfulness (does the answer contradict the source), Answer Relevance, Context Relevance.
- **LLM-as-a-judge** — using a frontier model to grade production output. Know the identity-bias problem, where a judge prefers its own style.
- **LangSmith** or **Arize Phoenix** — tracing every step of a long agentic loop to find where the agent called the wrong tool. Directly relevant to Custodian.
- **Guardrails AI / NeMo Guardrails** — preventing secret and PII leakage. You already built this yourself; knowing the standard tools lets you compare your approach to theirs, which is a strong interview move.

## 3. Coding rounds

- **LeetCode** — but only if you know a DSA round is coming. Ask in your conversation.
- **NeetCode 150** — the efficient path. Skip the 3,000-problem grind.
- **GeeksforGeeks** — best for Indian company-specific patterns.

## 4. System design

- **System Design Primer** (GitHub, free) — classic distributed systems.
- **ByteByteGo** — visual, paid, well-regarded.
- For AI specifically: practice designing something like a self-healing support agent that reads large doc sets, executes code to verify bugs, and files issues. The four layers to structure any answer around: **model, retrieval, orchestration, production** (eval, observability, guardrails, cost, deploy).

## 5. Company-specific and behavioural

- **AmbitionBox** and **Glassdoor** — actual Indian interview experiences and rounds, by company.
- **STAR method** for behavioural rounds. Custodian is your story for most of them.

## 6. Mocks

- **interviewing.io** and **Pramp** — free peer mocks.
- Cheapest option: explain Custodian out loud to a colleague. Talking is the skill that's actually being tested.

## Two important warnings

**Don't prep like it's 2023.** Spending your time on gradient descent and CNN architectures is called the single most common mistake, when roughly 75% of modern AI engineering interviews are about RAG, evaluation, and agentic systems.

**Some of these sites sell "interview co-pilot" tools** that feed you real-time answers during live interviews. Their written content is fine — skip the products. Getting caught ends the process, and it won't help you in the job.

Bookmark this. Then close it until you have a date.
