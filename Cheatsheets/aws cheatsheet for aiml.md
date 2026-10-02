# AWS AI/ML — In-Depth Cheatsheet

> Scope: the full AWS machine-learning stack as of 2026 — SageMaker AI (build/train/deploy), the managed AI services, generative AI (Bedrock, Q, Nova), supporting data services, core ML theory, security, and cost. Useful for the ML Engineer Associate (MLA-C01), ML Specialty (MLS-C01), AI Practitioner (AIF-C01), and day-to-day work.
>
> **Naming note (2024→2026):** The classic service for building/training/deploying models is now **Amazon SageMaker AI**. The name **Amazon SageMaker** now refers to a *unified platform* for data + analytics + AI, whose front door is **SageMaker Unified Studio** (with SageMaker Lakehouse and the Bedrock IDE). Fast-moving areas (Bedrock models, Q, Nova) change monthly — verify specifics against current AWS docs.

---

## 1. The AWS ML Stack — mental model

Three layers, bottom to top:

| Layer | What it is | Who uses it | Examples |
|---|---|---|---|
| **AI Services** | Pre-trained, API-driven. No ML expertise needed. | App developers | Rekognition, Comprehend, Transcribe, Polly, Translate, Textract, Lex, Kendra, Personalize |
| **ML Services (SageMaker AI)** | Build, train, tune, deploy your own models. | Data scientists / ML engineers | SageMaker AI (Studio, Autopilot, JumpStart, Pipelines, Feature Store…) |
| **ML Frameworks + Infrastructure** | Raw compute + DL frameworks. Full control. | ML researchers / platform teams | EC2 (P/G/Inf/Trn instances), Deep Learning AMIs/Containers, EKS, TensorFlow/PyTorch |
| **Generative AI** | Foundation models via API or in-account | Everyone | Bedrock, Amazon Q, Nova, SageMaker JumpStart FMs |

**Decision shortcut:** Prefer the *highest* layer that solves your problem. Only drop down when the AI service can't meet accuracy/customization needs (then SageMaker AI), or when you need full framework control (then raw infra).

---

## 2. Amazon SageMaker AI — the core

SageMaker AI covers the entire ML lifecycle: **prepare → build → train → tune → deploy → monitor → govern.**

### 2.1 Data preparation & labeling

| Capability | Purpose | Key points |
|---|---|---|
| **Data Wrangler** | Visual data prep | 300+ built-in transforms, data-quality/insights reports, exports to Pipelines/Feature Store. Fastest path from raw data to features. |
| **Processing Jobs** | Containerized pre/post-processing & evaluation | Run sklearn/Spark/custom containers for feature engineering or model eval, decoupled from training. |
| **Feature Store** | Central feature repository | **Online store** = low-latency reads for real-time inference; **Offline store** = S3-backed for training/batch. Ensures train/serve consistency, point-in-time correct joins. |
| **Ground Truth** | Data labeling | Human labeling (private workforce, vendors, or Mechanical Turk) + **automated labeling** (active learning labels easy items, humans do hard ones → cheaper). **Ground Truth Plus** = fully managed turnkey labeling. |
| **Clarify** | Bias detection + explainability | Pre-training bias metrics (e.g., class imbalance, DPL) and post-training metrics; **SHAP** values for feature attribution. Integrates with Model Monitor for bias/feature-attribution drift. |

### 2.2 Build & develop

- **SageMaker Studio** — browser IDE for the full lifecycle (notebooks, training, deployment, debugging).
- **SageMaker Unified Studio** — the newer unified data+analytics+AI environment (notebooks, visual ETL, SQL, Bedrock IDE for GenAI). SageMaker AI is a component within it.
- **Notebook instances** — classic managed Jupyter on a single EC2 instance.
- **JumpStart** — hub of pre-trained models (incl. foundation models), one-click deploy/fine-tune, and end-to-end solution templates.
- **Autopilot** — AutoML: give it tabular data + target column; it does feature eng, model selection, tuning, and produces an explainable leaderboard + notebooks.
- **Canvas** — no-code ML for business analysts (point-and-click predictions, integrates with Bedrock FMs and ready-to-use models).

### 2.3 Training — key knobs

**Instance families:**
| Family | Use | Notes |
|---|---|---|
| **ml.p** (P4/P5) | Heavy GPU training (DL) | Highest throughput, most expensive |
| **ml.g** (G5/G6) | Smaller GPU training/inference | Cost-effective GPU |
| **ml.trn** (Trainium) | Cost-optimized DL training | AWS custom silicon |
| **ml.c** | Compute-optimized CPU | Classic ML, inference |
| **ml.m** | General purpose | Light workloads |
| **ml.inf** (Inferentia) | Cost-optimized inference | AWS custom silicon |

**Input modes** (how data gets to the training container):
- **File mode** — copies full dataset to the instance before training (simple, default).
- **Pipe mode** — streams from S3 directly (no wait for full download; good for large data).
- **Fast File mode** — streams but presents files as if local (best of both; common default now).

**Cost / scale levers:**
- **Managed Spot Training** — up to ~90% savings; use **checkpointing** to S3 to survive interruptions.
- **Distributed training** — *Data Parallel* (SDP, split batches across GPUs) vs *Model Parallel* (SMP, split the model when it's too big for one GPU).
- **Warm pools** — keep infra alive between jobs to cut startup time.

**Training support tools:**
- **Automatic Model Tuning (HPO)** — strategies: **Bayesian** (smart, default), **Random**, **Grid**, **Hyperband** (early-stops bad trials, efficient). **Warm start** reuses prior tuning jobs.
- **Debugger** — captures tensors during training; detects vanishing/exploding gradients, overfitting, saturation; rules + alerts.
- **Experiments** — track runs, params, metrics for comparison/reproducibility.

### 2.4 Deployment — pick the right inference option

| Option | When | Characteristics |
|---|---|---|
| **Real-time endpoint** | Low-latency, steady traffic | Persistent, auto-scaling, always-on (you pay for it). |
| **Serverless Inference** | Spiky/intermittent traffic | Scales to zero; cold-start latency; you don't manage capacity. |
| **Asynchronous Inference** | Large payloads (≤1 GB), long processing | Queues requests, near-real-time, **can scale to zero**; good for big images/video/NLP docs. |
| **Batch Transform** | Offline scoring of whole datasets | No persistent endpoint; process in bulk then shut down. |

**Advanced patterns:**
- **Multi-Model Endpoints (MME)** — host many models behind one endpoint; load on demand → big cost savings when you have many small/infrequent models.
- **Multi-Container Endpoints** — multiple distinct containers on one endpoint (direct-invoke or as a pipeline).
- **Inference Pipelines** — chain 2–15 containers (e.g., preprocess → model → postprocess) as one endpoint.
- **Inference Recommender** — load-tests to right-size instance type/count.
- **Shadow testing / A-B (production variants)** — split traffic across model versions safely.
- **SageMaker Neo** — compile models to run faster/cheaper on target hardware (cloud or edge).
- **Edge Manager / IoT Greengrass** — deploy & manage models on edge devices.

### 2.5 MLOps & governance

| Service | Role |
|---|---|
| **Pipelines** | CI/CD for ML — define a DAG of steps (process → train → eval → register → deploy). The core MLOps orchestrator. |
| **Model Registry** | Version models, group into model packages, manage **approval status** (gate deployment). |
| **Projects** | MLOps templates wiring Pipelines + Registry + CI/CD (CodePipeline/CodeBuild). |
| **Model Monitor** | Detects **data-quality drift, model-quality drift, bias drift, feature-attribution drift** on live endpoints; captures requests for baseline comparison. |
| **Model Cards** | Document intended use, risk rating, metrics for governance/audit. |
| **ML Lineage Tracking** | Record entity lineage across the lifecycle for reproducibility/audit. |

**MLOps maturity shorthand:** manual notebooks → Pipelines automation → full CI/CD with Projects + Registry gating + Model Monitor feedback loop (retrain trigger on drift).

---

## 3. SageMaker built-in algorithms

Know *what each does* and *supervised vs unsupervised* — classic exam material.

### Supervised
| Algorithm | Problem | Notes |
|---|---|---|
| **Linear Learner** | Classification + regression | Trains many models in parallel, picks best; CSV or recordIO-protobuf. |
| **XGBoost** | Classification + regression | Gradient-boosted trees; most-used built-in; CSV/libsvm/Parquet; great tabular baseline. |
| **K-Nearest Neighbors (KNN)** | Classification + regression | Index-based; simple, non-parametric. |
| **Factorization Machines** | Classification/regression on **high-dimensional sparse** data | Recommendations, click prediction; recordIO-protobuf **float32 only**. |
| **DeepAR** | Time-series forecasting | RNN; learns across **many related series** (better than per-series ARIMA). |
| **Object2Vec** | Embeddings for arbitrary objects | General-purpose neural embeddings. |
| **Seq2Seq** | Sequence → sequence | Translation, summarization, speech-to-text. |
| **BlazingText** | Text classification + Word2Vec | Very fast, scalable embeddings. |

### Unsupervised
| Algorithm | Problem | Notes |
|---|---|---|
| **K-Means** | Clustering | Partition into k groups. |
| **PCA** | Dimensionality reduction | Compress features, remove correlation. |
| **Random Cut Forest (RCF)** | Anomaly detection | Unsupervised outlier scoring; also in Kinesis Analytics. |
| **IP Insights** | Anomaly detection on (entity, IPv4) pairs | Fraud/account-takeover signals. |
| **LDA** (Latent Dirichlet Allocation) | Topic modeling | CPU, single-instance. |
| **Neural Topic Model (NTM)** | Topic modeling | Neural, GPU-capable. |

### Computer vision / text
- **Image Classification**, **Object Detection (SSD)**, **Semantic Segmentation** — vision built-ins (ResNet/SSD backbones), support transfer learning.
- **Text Classification** via BlazingText; sequence tasks via Seq2Seq.

**Quick picks:**
- Tabular classification/regression → **XGBoost** (then Linear Learner).
- Recommendations / sparse → **Factorization Machines** (or the **Personalize** service).
- Forecasting many series → **DeepAR**.
- Anomaly detection → **RCF** (metrics/streams), **IP Insights** (entity+IP).
- Topic modeling → **LDA/NTM**.

---

## 4. Managed AI services

### Vision
| Service | What |
|---|---|
| **Rekognition** | Image/video: objects & scenes, face detection/analysis/comparison, celebrity & text in image, content moderation, PPE detection. **Custom Labels** to train on your own objects. |
| **Textract** | OCR++: extracts text, **forms (key-value)**, **tables**, and answers **Queries** from scanned docs/PDFs. |
| **Lookout for Vision** *(check current availability)* | Industrial defect detection from images. |

### Speech / language
| Service | What |
|---|---|
| **Transcribe** | Speech → text; speaker diarization, custom vocabulary, automatic language ID, **PII redaction**; **Transcribe Medical**; Call Analytics. |
| **Polly** | Text → speech; **neural (NTTS) voices**, **SSML** control, lexicons, speech marks. |
| **Translate** | Neural machine translation; **custom terminology**, real-time & batch. |
| **Comprehend** | NLP: entities, key phrases, sentiment, language detection, **PII detection**, syntax, topic modeling; **custom classification & entity recognition**; **Comprehend Medical** (PHI, ICD-10/RxNorm). |

### Conversational / search / recommendations
| Service | What |
|---|---|
| **Lex** | Chatbots/voice bots (same tech as Alexa). Core concepts: **intents**, **utterances**, **slots**, fulfillment via Lambda. |
| **Kendra** | Enterprise **intelligent (semantic) search** over your documents; connectors to S3, SharePoint, etc.; natural-language Q&A. |
| **Personalize** | Real-time recommendation engine (the Amazon.com recommender as a service); "user-personalization", "similar-items", "personalized-ranking" recipes. |
| **Q Business** | GenAI assistant over your enterprise data/apps (see §5). |

### Specialized / industrial / dev
| Service | What |
|---|---|
| **Fraud Detector** | Managed fraud scoring from your historical event data. |
| **Forecast** *(maintenance / check availability)* | Time-series forecasting service; AWS now often steers forecasting to **SageMaker Canvas / DeepAR**. |
| **Augmented AI (A2I)** | Insert **human review** workflows into ML predictions (low-confidence routing). |
| **HealthLake / Comprehend Medical / Transcribe Medical** | Healthcare NLP + FHIR data store. |
| **CodeGuru** | ML-powered code reviews (Reviewer) + runtime profiling (Profiler). |
| **DevOps Guru** | ML detection of operational anomalies in your AWS ops data. |
| **Lookout for Equipment / Monitron** | Industrial equipment anomaly detection (sensors). |
| **Panorama** | Computer vision at the edge on-prem cameras. |
| **DeepRacer** | 1/18 RL race car — learning/gamified reinforcement learning. |

> Several narrow services (Forecast, Lookout for Metrics, DeepComposer, DeepLens) have been placed in maintenance or discontinued over time. Confirm current status before building on them.

---

## 5. Generative AI on AWS

### 5.1 Amazon Bedrock — managed foundation-model platform
Fully managed, **serverless** access to **100+ FMs from ~18 providers** through one API (Anthropic **Claude**, Amazon **Nova/Nova 2**, Meta **Llama**, Mistral, Cohere, AI21, Stability, and others). Switch models by changing an ID, not re-architecting.

**Core building blocks:**
| Feature | Purpose |
|---|---|
| **Converse API** | Unified multi-turn chat interface across models. |
| **Knowledge Bases** | Fully managed **RAG**: auto chunking, embeddings, vector indexing; `Retrieve` / `RetrieveAndGenerate` APIs with source attribution. Vector stores: OpenSearch Serverless, Aurora/pgvector, etc. |
| **Agents** | Multi-step task orchestration — call APIs/Lambda ("action groups"), query Knowledge Bases, retain session memory. |
| **AgentCore** | Managed runtime/infrastructure for **production-grade agents** at scale. |
| **Guardrails** | Content filters, denied topics, **PII redaction**, word filters, and contextual-grounding/hallucination checks. Apply across models. |
| **Flows** | Visual chaining of prompts, models, KBs, and logic into workflows. |
| **Model Evaluation** | Automatic + human eval to compare models on your tasks. |
| **Customization** | **Fine-tuning** and **continued pre-training** with your data (private copy of the model). |
| **Provisioned Throughput** | Reserved capacity for steady/high-volume + required for some custom models. |
| **Prompt Management / Caching / Intelligent Prompt Routing** | Reusable prompts, cached tokens for cost, auto-route to cheapest adequate model. |

**Pricing modes:** on-demand (per-token), batch, and provisioned throughput. Nova Lite is cheap (~$0.06/M input tokens); frontier Claude/others cost more.

### 5.2 Amazon Nova
Amazon's own FM family on Bedrock: **Micro / Lite / Pro / Premier** (text), **Canvas** (images), **Reel** (video), **Sonic** (speech-to-speech). The **Nova 2** generation (Lite/Pro/Omni/Sonic) adds stronger reasoning, agentic, and multimodal capability.

### 5.3 Amazon Q
- **Q Developer** — AI coding assistant (IDE/CLI/console); code gen, debugging, transformations/upgrades, AWS expertise; embedded in SageMaker Unified Studio.
- **Q Business** — GenAI assistant grounded on your enterprise data with connectors + access controls.

### 5.4 SageMaker JumpStart (GenAI side)
Deploy/fine-tune foundation models **in your own account/VPC** (vs Bedrock's fully managed API) — more control, data stays in your account.

**Bedrock vs JumpStart:** Bedrock = serverless, pay-per-use, managed, fastest to build RAG/agents. JumpStart = you run the model on your infra inside your VPC, more customization/control.

### 5.5 RAG vs fine-tuning (quick guide)
- **RAG (Knowledge Bases)** → inject *current/proprietary knowledge*; cheaper, no retraining, easy to update, reduces hallucination.
- **Fine-tuning** → change *behavior/style/format* or teach a narrow task; needs labeled data + cost.
- **Prompt engineering** → always try first; cheapest lever.
- Combine: prompt + RAG for knowledge, fine-tune only if behavior still off.

---

## 6. Supporting data & analytics services

| Service | ML role |
|---|---|
| **S3** | The data lake for all ML data, model artifacts, logs. Default storage. |
| **AWS Glue** | Serverless ETL + **Data Catalog** (schema/metadata); Glue DataBrew = visual cleaning. |
| **Athena** | Serverless SQL over S3 (query training data in place). |
| **EMR** | Managed Spark/Hadoop for big-data feature engineering at scale. |
| **Kinesis Data Streams / Firehose / Managed Flink** | Real-time ingest/streaming; **Flink supports Random Cut Forest** for streaming anomaly detection. |
| **Redshift / Redshift ML** | Warehouse; **Redshift ML** = create/train models with SQL (`CREATE MODEL`, backed by SageMaker Autopilot). |
| **Aurora ML / Athena ML** | Invoke SageMaker/Comprehend from SQL. |
| **SageMaker Lakehouse** | Iceberg-compatible unified data layer (S3/Redshift) under the new SageMaker platform. |
| **QuickSight** | BI dashboards; **QuickSight Q** = natural-language analytics; ML Insights (anomaly/forecast). |
| **OpenSearch Serverless** | Common vector store for Bedrock Knowledge Bases / semantic search. |

---

## 7. Core ML theory (exam staples)

### Evaluation metrics
**Classification:**
- **Confusion matrix** → TP, FP, TN, FN.
- **Precision** = TP/(TP+FP) — "when it says positive, how often right." Use when **false positives are costly** (e.g., spam flagging good email).
- **Recall / Sensitivity / TPR** = TP/(TP+FN) — "of actual positives, how many caught." Use when **false negatives are costly** (e.g., disease, fraud).
- **F1** = harmonic mean of precision & recall — balance, good for **imbalanced** classes.
- **Accuracy** — misleading on imbalanced data.
- **ROC curve / AUC** — threshold-independent ranking quality; AUC 0.5 = random, 1.0 = perfect.
- **Specificity / TNR** = TN/(TN+FP).

**Regression:** MAE, **MSE/RMSE** (penalizes large errors), **R²** (variance explained), MAPE.

### Bias–variance & fit
- **Overfitting** = low train error, high test error (high variance). Fixes: more data, **regularization (L1/L2)**, dropout, early stopping, simpler model, cross-validation.
- **Underfitting** = high train & test error (high bias). Fixes: more features, more complex model, train longer.
- **L1 (Lasso)** → sparse, feature selection. **L2 (Ridge)** → shrinks weights, keeps all features.

### Data handling
- **Imbalanced classes:** oversample minority (**SMOTE**), undersample majority, class weights, choose F1/AUC over accuracy.
- **Missing values:** drop, impute (mean/median/mode, model-based).
- **Scaling:** normalization (0–1) vs standardization (z-score) — needed for distance/gradient methods (KNN, K-Means, NN), not for trees.
- **Encoding:** one-hot (low cardinality), label/ordinal, target/embedding (high cardinality).
- **Splits:** train / validation / test; **k-fold cross-validation** for small data; beware leakage (fit scalers on train only).

### Neural-net knobs
- **Learning rate** (most important), batch size, epochs.
- **Activations:** ReLU (hidden default), sigmoid (binary out), softmax (multiclass out), tanh.
- **Vanishing/exploding gradients** → ReLU, batch norm, gradient clipping, residual connections (Debugger can detect).
- **Optimizers:** Adam (default), SGD + momentum.

---

## 8. Security, privacy & responsible AI

- **IAM** — least-privilege roles; SageMaker execution roles scope what jobs/endpoints can access.
- **VPC / PrivateLink** — run training/endpoints in your VPC with no internet; VPC endpoints to reach S3/Bedrock privately.
- **Network isolation** — training/inference containers with no outbound network (for untrusted code/data).
- **Encryption** — **KMS** at rest (S3, EBS, model artifacts) + TLS in transit; `VolumeKmsKeyId`, `OutputKmsKeyId`.
- **Macie** — discover/classify PII in S3 (pre-training hygiene).
- **PII handling** — Comprehend PII detection, Transcribe redaction, Bedrock Guardrails PII masking.
- **Responsible AI** — **Clarify** (bias + explainability), **Model Cards**, **Model Monitor** (drift/bias drift), Bedrock **Guardrails** + **Model Evaluation**, **A2I** human review.
- **Governance** — SageMaker Role Manager, Model Registry approval gates, CloudTrail audit, lineage tracking.
- **Data residency** — JumpStart/Bedrock customization keep data in your account/region; your prompts/data aren't used to train base models.

---

## 9. Cost optimization

- **Spot training** (checkpoint to S3) — up to ~90% off.
- **Right-size** with Inference Recommender; use **Inferentia (inf)** / **Trainium (trn)** silicon.
- **Serverless / Async inference** to **scale to zero** on spiky or batch workloads.
- **Multi-Model Endpoints** to consolidate many models.
- **Batch Transform** instead of a live endpoint for offline scoring.
- **Auto-scaling** on real-time endpoints; delete idle endpoints/notebooks (biggest silent cost).
- **Savings Plans** for steady SageMaker usage.
- GenAI: prompt caching, intelligent prompt routing, batch inference, and picking the smallest adequate model (e.g., Nova Lite vs frontier).

---

## 10. "Which service?" quick-reference

| Need | Reach for |
|---|---|
| Extract text/tables/forms from scanned docs | **Textract** |
| Detect objects/faces/moderation in images | **Rekognition** |
| Sentiment/entities/PII from text | **Comprehend** |
| Transcribe audio to text | **Transcribe** |
| Natural-sounding voice output | **Polly** |
| Translate languages | **Translate** |
| Build a chatbot | **Lex** (+ Bedrock for GenAI chat) |
| Enterprise document search / Q&A | **Kendra** (or Bedrock Knowledge Bases) |
| Product/content recommendations | **Personalize** |
| No-code predictions for analysts | **SageMaker Canvas** |
| AutoML on tabular data | **Autopilot** (or Redshift ML via SQL) |
| Custom DL model, full control | **SageMaker AI** training + endpoints |
| Call an LLM via API (Claude/Nova/Llama) | **Bedrock** |
| Managed RAG over my docs | **Bedrock Knowledge Bases** |
| Multi-step AI agent | **Bedrock Agents / AgentCore** |
| Run/fine-tune an FM inside my VPC | **SageMaker JumpStart** |
| AI coding assistant | **Amazon Q Developer** |
| Human-in-the-loop review of predictions | **Augmented AI (A2I)** |
| Streaming anomaly detection | **Kinesis/Flink + RCF** |
| Forecast many time series | **DeepAR** / Canvas |

---

### Highest-yield facts to memorize
1. **Inference option → trait:** Real-time (steady/low-latency), Serverless (spiky, scale-to-zero), Async (large payloads ≤1 GB, scale-to-zero), Batch (offline whole datasets). MME = many models/one endpoint.
2. **Precision vs recall:** FP-costly → precision; FN-costly → recall; imbalanced → F1/AUC.
3. **Input modes:** File / Pipe / Fast File.
4. **HPO strategies:** Bayesian, Random, Grid, Hyperband (early stop).
5. **RAG vs fine-tune:** knowledge → RAG; behavior/format → fine-tune; try prompting first.
6. **Clarify = bias + explainability (SHAP); Model Monitor = drift (data/model/bias/feature-attribution).**
7. **Spot + checkpointing** for cheap training; **Inferentia/Trainium** for cheap silicon.
8. **XGBoost** = go-to tabular; **DeepAR** = many time series; **RCF** = anomalies; **Factorization Machines** = sparse/recs.
9. **Bedrock building blocks:** Knowledge Bases (RAG), Agents/AgentCore, Guardrails, Flows, Model Evaluation, Provisioned Throughput.
10. **Encryption (KMS) + VPC + IAM least-privilege + network isolation** = the security quartet.
