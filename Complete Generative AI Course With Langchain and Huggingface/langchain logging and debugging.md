  # LangChain Logging & Debugging

LangChain manages logging through a combination of built-in flags, an event-driven callback system, and direct integrations with dedicated LLM observability platforms. The framework categorizes logging tools into native features, internal callbacks, and external integrations to help debug chains, agents, and prompts effectively.

---

## Built-In Framework Tools

- **`LANGCHAIN_VERBOSE`** — An environment variable flag. Prints critical intermediate chain steps directly to your console.
- **`set_debug(True)`** — A global framework configuration. Forces LangChain to output exhaustive input/output logs for every single nested component.
- **Python Native `logging`** — Standard library integration. You can set levels (`DEBUG`, `INFO`) or route system logs to custom files or standard streams via framework setup.

---

## The Callback System

The `callbacks` argument is the core mechanism for custom logging in LangChain. It lets you hook into specific execution states using standard or customized handlers.

- **`StdOutCallbackHandler`** — Built-in handler that prints all text logs straight to standard terminal output.
- **`FileCallbackHandler`** — Directs pipeline events and raw component responses to an active local text file.
- **Custom Callbacks** — Inherit from `BaseCallbackHandler` to write your own hooks, enabling structured formats like JSON logs for integration with tools like Datadog, Splunk, or Elasticsearch.

---

## First-Party Observability

- **[LangSmith](https://www.langchain.com/langsmith/observability)** — LangChain's official UI-driven platform. Tracks entire agent trajectories, token usage, latency, and exact nested tool payloads. Activate it globally with `LANGSMITH_TRACING=true`.

---

## Third-Party Partner Tools

LangChain features pre-built integrations for external logging and tracking suites:

- **[MLflow](https://mlflow.org/docs/latest/genai/flavors/langchain/autologging/)** — Uses `mlflow.langchain.autolog()` to automatically capture model signatures, metrics, and application traces into an MLflow instance.
- **Portkey** — Directs production API logging to a centralized dashboard while offering automatic retries and semantic caching.
- **[Pydantic Logfire](https://docs.langchain.com/langsmith/trace-with-opentelemetry)** — Integrates over OpenTelemetry to deliver clean, structured data logging optimized for Pydantic data schemas.
- **Log10** — A proxiless data management platform built to instantly log, tag, and organize every individual LangChain model call.

---

## Quick Reference

| Category | Tool | Purpose |
| --- | --- | --- |
| Built-in flag | `LANGCHAIN_VERBOSE` | Console output of key chain steps |
| Built-in flag | `set_debug(True)` | Exhaustive nested I/O logs |
| Built-in | Python `logging` | Standard levels, custom sinks |
| Callback | `StdOutCallbackHandler` | Logs to terminal |
| Callback | `FileCallbackHandler` | Logs to local file |
| Callback | Custom `BaseCallbackHandler` | Structured/JSON logs for SIEM tools |
| First-party | LangSmith | Full-trajectory tracing UI |
| Third-party | MLflow, Portkey, Logfire, Log10 | External observability suites |
