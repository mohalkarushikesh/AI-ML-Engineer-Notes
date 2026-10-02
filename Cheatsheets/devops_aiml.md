# DevOps for AI/ML (MLOps) — In-Depth Cheatsheet

> Scope: DevOps as it applies to machine learning — the practices, pipeline, and tooling that take a model from notebook to reliable production and keep it healthy. Concepts are tool-agnostic; the tool names reflect the current landscape (versions/tools evolve, so verify specifics). Includes the LLMOps layer for generative AI.

---

## 1. Why ML DevOps ≠ traditional DevOps

Traditional software = **code**. ML systems = **code + data + model**. That third-and-fourth axis breaks classic assumptions.

| Dimension | Traditional DevOps | MLOps |
|---|---|---|
| Artifacts | Code | Code + **data** + **model weights** + config |
| Versioning | Git on code | Git **plus** data versioning + model/experiment versioning |
| Testing | Unit/integration | Plus **data validation** + **model behavioral tests** |
| "Build" output | Binary/image | Trained model (non-deterministic, depends on data) |
| Decay | Code is static once correct | Models **degrade over time** (drift) → need retraining |
| Extra pipeline stage | CI + CD | CI + CD + **CT (Continuous Training)** |
| Reproducibility | Code + deps | Code + deps + **data snapshot + random seed + env + hyperparams** |
| Infra | CPU, stateless | GPUs/accelerators, large artifacts, stateful data stores |
| Monitoring | Latency, errors, uptime | Plus **data drift, concept drift, prediction quality** |

**The core insight:** a passing test suite and green deploy don't mean the model is *correct* — correctness depends on live data that changes after deploy. MLOps exists to close that loop.

---

## 2. MLOps maturity levels (Google's canonical model)

| Level | Name | What it means | Signals |
|---|---|---|---|
| **0** | Manual process | Notebooks, manual handoff to ops, no CI/CD, infrequent releases | Data scientist emails a pickle file; ops wraps it by hand |
| **1** | ML pipeline automation | Automated, orchestrated training pipeline; **Continuous Training** triggered by new data/drift; feature store; pipeline reproducible | Model retrains automatically; same pipeline runs in dev & prod |
| **2** | CI/CD pipeline automation | Full automation of the *pipeline itself*: build/test/deploy the pipeline via CI/CD, automated model deployment with gating | Push code → tests → build pipeline → deploy → monitor, hands-off |

(Microsoft's model goes 0–4 adding "repeatable" and "fully automated/optimized" tiers; same spirit.)

**Progression goal:** move the human from *running steps* to *reviewing and approving* — automate the toil, keep judgment at the gates.

---

## 3. The end-to-end ML pipeline (CI / CD / CT)

```
          ┌─────────── Continuous Training (CT) loop ───────────┐
          ▼                                                     │
Data → Validate → Feature Eng → Train → Evaluate → Register → Deploy → Monitor
          │           │            │        │          │          │        │
       schema       feature      HPO/     gate on    version   canary/   drift/
       checks       store        track    metrics    + approve shadow    perf alerts
```

- **CI (Continuous Integration):** lint, unit tests, **data validation**, **model tests**, build training container.
- **CD (Continuous Delivery/Deployment):** deploy the *pipeline* and/or the *model* (as a service), with progressive rollout + automatic rollback.
- **CT (Continuous Training):** retrain automatically on triggers — schedule, new data volume, or **detected drift / performance drop**.

**Triggers for retraining:** calendar (e.g., weekly), data-volume threshold, drift alarm, performance degradation, or on-demand.

---

## 4. Versioning — the four things you must track

Reproducibility requires pinning **all** of these together:

| What | Tool(s) | Notes |
|---|---|---|
| **Code** | Git | Standard; branch per experiment/feature. |
| **Data** | **DVC**, lakeFS, Git-LFS, Delta Lake time-travel | DVC stores lightweight pointers in Git, data in remote (S3/GCS); `dvc.yaml` defines reproducible pipeline stages. lakeFS = git-like branches over object storage. |
| **Experiments** | **MLflow Tracking**, **Weights & Biases**, Neptune, Comet | Log params, metrics, artifacts, code version per run; compare runs. |
| **Models** | **MLflow Model Registry**, W&B Registry, SageMaker Model Registry | Versioned models + stage transitions (Staging → Production) + approval. |
| **Environment** | Docker image digest, `requirements.txt`/`poetry.lock`/conda, Nix | Pin exact dependency versions + base image. |

**Golden rule:** a model version is reproducible only if you can recover *(data snapshot + code commit + config/hyperparams + env image + seed)*. Log all five together.

---

## 5. Containerization — Docker for ML

- **Dockerfile** packages code + deps + model into a portable image; the unit of deployment.
- **Multi-stage builds** → small runtime images (build deps in stage 1, copy only artifacts to stage 2).
- **GPU in containers** → NVIDIA Container Toolkit; base on `nvidia/cuda` or framework images (`pytorch/pytorch`, `tensorflow/tensorflow`); match CUDA/cuDNN to the framework.
- **Layer caching** → put rarely-changing steps (deps install) *before* frequently-changing ones (code copy) for fast rebuilds.
- Keep images lean: slim base, `.dockerignore`, no training data baked in, pin versions.
- **Registries:** Docker Hub, ECR, GCR/Artifact Registry, GHCR.

---

## 6. Orchestration — pipelines & Kubernetes

### Pipeline/workflow orchestrators
| Tool | Best for | Notes |
|---|---|---|
| **Apache Airflow** | General DAG scheduling | Mature, huge operator ecosystem; task-centric. |
| **Kubeflow Pipelines (KFP)** | ML pipelines on Kubernetes | Container-native steps; built on **Argo Workflows**. |
| **Argo Workflows** | K8s-native workflows | YAML DAGs; the engine under KFP. |
| **Prefect** | Pythonic, dynamic flows | Modern, good local→cloud story. |
| **Dagster** | Data-asset-aware pipelines | Models pipelines as **assets** with lineage + types. |
| **Metaflow** | DS-friendly (Netflix) | Minimal boilerplate, scales to cloud. |

### Kubeflow ecosystem (ML on K8s)
- **Pipelines (KFP)** — orchestration.
- **Katib** — hyperparameter tuning / NAS.
- **Training Operators** — distributed PyTorch/TF/XGBoost jobs.
- **KServe** — serverless model serving (autoscale to zero, canary, explainers).
- **Notebooks** — managed Jupyter.

### Helm
- K8s package manager; **charts** template your deployments; manage releases/rollbacks. Common for deploying serving stacks.

---

## 7. Model serving & deployment

### Serving patterns
| Pattern | When | Interface |
|---|---|---|
| **Online / real-time** | Low-latency per-request | REST / **gRPC** |
| **Batch** | Score large datasets offline | Job reads/writes storage |
| **Streaming** | Event-driven scoring | Kafka/Kinesis + consumer |
| **Embedded / edge** | On-device, offline, low latency | Model compiled into app (ONNX, TFLite, CoreML) |

**Model-as-service** (separate deployable, independent scaling) vs **embedded** (model shipped inside the app). Service = flexible/scalable; embedded = simple/low-latency.

### Serving tools
| Tool | Strength |
|---|---|
| **NVIDIA Triton** | Multi-framework, GPU, **dynamic batching**, model ensembles, concurrent models. |
| **TorchServe / TF Serving** | Framework-native serving. |
| **BentoML** | Package "Bentos" (model + code + deps) → containerized API; adaptive batching. |
| **Seldon Core / KServe** | K8s-native, canary, explainers, A/B, scale-to-zero. |
| **Ray Serve** | Python-native, compose multiple models, scalable. |
| **vLLM / TGI** | High-throughput **LLM** serving (paged attention, continuous batching). |

### Deployment strategies (progressive delivery)
| Strategy | How | Use |
|---|---|---|
| **Recreate** | Stop old, start new | Dev/simple; downtime OK. |
| **Rolling** | Replace instances gradually | Default zero-downtime. |
| **Blue-Green** | Two full envs, flip traffic | Instant rollback; costs 2×. |
| **Canary** | Send small % to new model, ramp up | Limit blast radius; watch metrics. |
| **Shadow (dark launch)** | New model gets copy of live traffic, responses discarded | **Test on real traffic with zero user risk.** |
| **A/B test** | Split users, compare business KPIs | Validate model *impact*, not just accuracy. |
| **Multi-armed bandit** | Dynamically route more traffic to the winner | Optimize while testing. |

---

## 8. Infrastructure as Code (IaC)

| Tool | Scope |
|---|---|
| **Terraform** | Cloud-agnostic provisioning (HCL); providers for AWS/GCP/Azure/K8s; **state** tracks real infra; `plan`→`apply`. |
| **Pulumi** | IaC in real languages (Python/TS). |
| **CloudFormation / CDK** | AWS-native IaC (CDK = code-defined). |
| **Ansible** | Configuration management / provisioning. |
| **Helm / Kustomize** | K8s manifests templating/overlays. |

**Principles:** everything reproducible from code, no click-ops in prod, immutable infra, environment parity (dev≈staging≈prod), GitOps (Git is the source of truth; **Argo CD / Flux** sync cluster to repo).

---

## 9. CI/CD tooling

| Tool | Notes |
|---|---|
| **GitHub Actions** | YAML workflows, huge marketplace; common default. |
| **GitLab CI** | Integrated pipelines + registry. |
| **Jenkins** | Self-hosted, plugin-rich, mature. |
| **CircleCI / Buildkite / Azure DevOps** | Alternatives. |
| **CML (Continuous ML)** | Iterative.ai — posts metrics/plots as PR comments, provisions cloud runners for training in CI. |
| **Argo CD / Flux** | GitOps CD for Kubernetes. |

**ML CI stages:** lint → unit tests → **data validation** → train (or smoke-train) → **model eval + behavioral tests** → build image → push registry → deploy (canary) → monitor.

---

## 10. Monitoring & observability

### Three pillars (classic) + ML additions
- **Metrics, Logs, Traces** (classic observability) — Prometheus (metrics) + Grafana (dashboards); ELK/Loki (logs); OpenTelemetry/Jaeger (traces).
- **ML-specific signals:**
  - **Data quality** — nulls, ranges, schema, cardinality.
  - **Data drift (covariate shift)** — input distribution changes vs training.
  - **Concept drift** — the X→y relationship changes (hardest; needs labels).
  - **Prediction drift** — output distribution shifts.
  - **Model performance** — accuracy/precision/recall over time (needs ground-truth labels, often delayed).

### Drift types (know the distinctions)
| Type | What changes | Example |
|---|---|---|
| **Covariate / data drift** | P(X) | New user demographics |
| **Concept drift** | P(y\|X) | Fraud tactics evolve; same inputs now mean something else |
| **Label / prior shift** | P(y) | Class balance changes |
| **Prediction drift** | model outputs | Early warning when labels are delayed |

**Detection methods:** PSI (Population Stability Index), KL divergence, KS test, Chi-square, Wasserstein distance, model-based (train a classifier to tell train-vs-prod apart).

### ML monitoring tools
**Evidently AI** (open-source drift/quality reports), **WhyLabs**, **Arize**, **Fiddler**, **Aporia**, **NannyML** (performance estimation without labels). Pair with Prometheus/Grafana for infra.

**Alerting + action:** drift alarm → trigger retraining pipeline (closes the CT loop) and/or page on-call.

---

## 11. Testing for ML (the testing "pyramid" extended)

| Layer | Tests | Tools |
|---|---|---|
| **Code** | Unit tests of feature/transform/serving code | pytest |
| **Data** | Schema, ranges, nulls, distribution, freshness | **Great Expectations**, **TensorFlow Data Validation (TFDV)**, Pandera, Pydantic |
| **Model behavioral** | **Invariance** (irrelevant perturbation ⇒ same output), **Directional** (known change ⇒ expected direction), **Minimum functionality** (per-slice must-pass cases) | CheckList-style tests, deepchecks |
| **Model quality gates** | Metric thresholds; compare vs current prod ("champion/challenger") | eval scripts in CI |
| **Integration / pipeline** | End-to-end pipeline runs on sample data | pipeline framework test modes |
| **Infra / load** | Latency, throughput, autoscaling | k6, Locust |

**Key idea:** don't just assert `accuracy > X` — test *slices* (subgroups), *robustness*, and *no regression vs production*.

---

## 12. Feature stores

Solve **train/serve skew** and feature reuse.
- **Offline store** — historical features for training (warehouse/S3); supports **point-in-time correct** joins (no label leakage).
- **Online store** — low-latency features for real-time inference (Redis/DynamoDB).
- **Registry** — feature definitions + lineage + discovery.
- Tools: **Feast** (open-source), Tecton, SageMaker Feature Store, Databricks Feature Store.

---

## 13. LLMOps — DevOps for generative AI

The MLOps loop plus new concerns specific to foundation models:

| Concern | Practice / tools |
|---|---|
| **Prompt management** | Version prompts like code; prompt registries; A/B prompts. (LangSmith, Langfuse, PromptLayer, Agenta) |
| **RAG pipelines** | Chunking, embeddings, **vector DBs** (Pinecone, Weaviate, Qdrant, Milvus, Chroma, pgvector), retrieval eval. |
| **Evaluation** | **LLM-as-judge**, reference-based + reference-free; **RAGAS** (RAG metrics: faithfulness, relevance), **DeepEval**, **promptfoo**, TruLens. Build golden eval sets. |
| **Observability / tracing** | Trace chains/agents end-to-end: **LangSmith, Langfuse, Phoenix (Arize), Helicone**. |
| **Guardrails** | Input/output filtering, PII, jailbreak defense: **NeMo Guardrails, Guardrails AI**, provider guardrails (e.g., Bedrock Guardrails). |
| **Cost & latency** | Token accounting, **semantic caching**, model routing (cheap model first), batching. |
| **Serving** | **vLLM, TGI**, quantization (GPTQ/AWQ), KV-cache, continuous batching. |
| **Fine-tuning ops** | LoRA/QLoRA adapters as versioned artifacts; eval-gated promotion. |
| **Drift for LLMs** | Output quality drift, hallucination rate, user-feedback (thumbs), eval-set regressions. |

**Rule of thumb:** prompt engineering → RAG → fine-tuning, in that order of cost; add evals + guardrails + tracing before going to production.

---

## 14. Security & governance

- **Secrets** — never in images/Git; use Vault, AWS Secrets Manager, SSM, sealed-secrets.
- **Supply chain** — scan images (Trivy, Grype), pin base images, **SBOM**, sign artifacts (Sigstore/cosign); beware malicious model files (prefer **safetensors** over pickle).
- **Model security** — adversarial robustness, prompt injection (LLMs), model/ data exfiltration, access control on endpoints.
- **Data governance** — PII handling, lineage, access policies, retention; label leakage prevention.
- **Reproducibility & audit** — lineage tracking, model cards, approval gates in registry, immutable logs.
- **Compliance** — document datasets, bias testing, explainability (SHAP/LIME), human-in-the-loop for high-stakes decisions.

---

## 15. Best practices & anti-patterns

**Do**
- Automate the pipeline, keep humans at **approval gates**, not step-running.
- Make everything reproducible (data + code + config + env + seed).
- Test data and model behavior, not just code.
- Monitor drift *and* business metrics; wire alarms to retraining.
- Use a feature store (or shared feature code) to kill train/serve skew.
- Canary/shadow every model deploy; keep instant rollback.
- Version and register models with stage + approval.

**Avoid (common anti-patterns)**
- "Throw the pickle over the wall" to ops.
- Training in notebooks that can't be re-run deterministically.
- No monitoring → silent model decay ("set and forget").
- Train/serve skew from duplicated feature logic.
- Deploying on accuracy alone (ignoring slices, drift, latency, cost).
- Manual, click-ops infrastructure with no IaC.
- Treating a green CI as "model is correct."

---

## 16. Tool landscape at a glance

| Category | Tools |
|---|---|
| Data versioning | DVC, lakeFS, Delta Lake, Git-LFS |
| Experiment tracking | MLflow, Weights & Biases, Neptune, Comet |
| Model registry | MLflow Registry, W&B, SageMaker |
| Orchestration | Airflow, Kubeflow Pipelines, Argo, Prefect, Dagster, Metaflow |
| Feature store | Feast, Tecton, SageMaker Feature Store |
| Containers / infra | Docker, Kubernetes, Helm, Terraform, Pulumi |
| Serving | Triton, TorchServe, TF Serving, BentoML, KServe, Seldon, Ray Serve, vLLM, TGI |
| CI/CD | GitHub Actions, GitLab CI, Jenkins, CML, Argo CD, Flux |
| Data/model testing | Great Expectations, TFDV, Pandera, deepchecks |
| Monitoring / drift | Evidently, WhyLabs, Arize, Fiddler, NannyML, Prometheus+Grafana |
| LLMOps | LangSmith, Langfuse, RAGAS, DeepEval, promptfoo, NeMo Guardrails; vector DBs: Pinecone/Weaviate/Qdrant/Milvus/Chroma/pgvector |

---

### Highest-yield takeaways
1. **ML = code + data + model** → version all four (+ env + seed) or it isn't reproducible.
2. **CI + CD + CT** — Continuous Training is the ML-specific pipeline stage; drift triggers it.
3. **MLOps maturity:** 0 manual → 1 pipeline automation → 2 CI/CD of the pipeline itself.
4. **Train/serve skew** is the classic silent bug → feature stores / shared feature code.
5. **Drift types:** covariate (P(X)), concept (P(y|X)), label (P(y)), prediction — detect with PSI/KS/KL.
6. **Deploy progressively:** shadow (zero-risk real traffic) → canary → A/B; keep rollback.
7. **Test data + model behavior**, not just code; gate on no-regression vs production.
8. **Monitoring closes the loop:** drift alarm → retrain → re-evaluate → redeploy.
9. **LLMOps adds:** prompt versioning, RAG + vector DBs, evals (RAGAS/LLM-judge), guardrails, token-cost + tracing.
10. **Security:** safetensors over pickle, scan/sign images, secrets out of Git, lineage + approval gates.
