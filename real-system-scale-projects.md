Fair — those were still portfolio-tier. Here are two that operate at real-system scale, where the difficulty is in the engineering around the model, not the model call.

**1. Self-improving agentic research system**

An autonomous system that takes an open-ended research question, decomposes it into sub-questions, runs parallel investigation threads (web + internal corpus), critiques its own intermediate findings, and synthesizes a cited report — then evaluates its own output and refines.

What makes it next-level:
- **Orchestration under uncertainty** — planner/worker/critic architecture with dynamic re-planning when a thread hits a dead end, not a fixed pipeline.
- **Memory across the run** — a working memory that dedupes findings, tracks which claims are supported vs. speculative, and resolves contradictions between sources.
- **Self-evaluation loop** — an LLM-as-judge harness plus a factuality checker that scores the report and triggers targeted re-investigation of weak sections.
- **Cost and latency budgeting** — token/time budgets enforced across a tree of dozens of model calls, with graceful degradation when the budget runs low.

The hard, employable part isn't the prompting — it's making a system with 50+ interdependent LLM calls reliable, observable, and debuggable. Build a trace viewer so you can see why any given run failed.

**2. Production LLM inference platform with continuous evaluation**

A serving system that sits in front of one or more models and handles the concerns that actually break in production: caching, routing, guardrails, and a regression harness that catches quality drops before users do.

What makes it next-level:
- **Semantic caching** — cache by embedding similarity, not exact match, with correctness safeguards against serving a near-but-wrong cached answer.
- **Model routing** — route easy queries to a cheap/fast model and hard ones to a strong model, using a lightweight classifier, and measure the cost/quality tradeoff empirically.
- **Guardrail layer** — input/output filtering for prompt injection, PII, and jailbreaks, with the security thinking that goes into it.
- **Continuous eval in CI** — a golden dataset that runs on every prompt or model change, flagging regressions automatically. This is the piece almost no portfolio has, and it's exactly what companies are desperate for.
- **Observability** — latency percentiles, token spend per route, cache hit rate, and quality scores on a live dashboard.

---

Both are systems you could genuinely run for real users, and both demonstrate the thing that's rare: engineering discipline around unreliable models. If you pick one, tell me your stack and whether you have GPU access and I'll give you an architecture diagram and a milestone-based build plan.
